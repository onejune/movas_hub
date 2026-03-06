import torch
import torch.nn.functional as F
import random

class GradNormManager:
    """
    Manages the GradNorm algorithm for balancing losses in multi-task models like MMoE.
    """

    def __init__(self, model, task_weights, num_tasks, alpha, update_frequency=5):
        """
        Initializes the GradNorm manager.

        Args:
            model: The PyTorch model (e.g., MtlMMoEModel).
            task_weights: nn.Parameter of shape (num_tasks,) representing learnable weights.
            num_tasks: Number of tasks in the model.
            alpha: Learning rate for the task weights optimization.
            update_frequency: How often to update the weights (every N steps).
        """
        self.model = model
        self.task_weights = task_weights
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.step_counter = 0

        # Optimizer for task weights
        self.optimizer = torch.optim.Adam([self.task_weights], lr=self.alpha)

        # Storage for initial losses
        self.initial_losses = None
        self.initial_losses_recorded = False

        # Get shared parameters (those before task-specific towers)
        self.shared_params = list(model.get_shared_params())

        print(f"[GradNorm Manager.__init__] alpha: {self.alpha}, update_frequency: {self.update_frequency}, task_weights: {self.task_weights.tolist()}")
        print(f"[GradNorm Manager.__init__] Shared params: {self.shared_params}")

    def record_initial_losses(self, losses):
        """Records the initial losses for each task."""
        if not self.initial_losses_recorded:
            self.initial_losses = torch.tensor(losses, dtype=torch.float32, device=self.task_weights.device)
            self.initial_losses_recorded = True
            print(f"[GradNorm Manager] Initial losses recorded: {self.initial_losses.tolist()}")

    def compute_weighted_losses(self, individual_losses):
        """Computes the weighted version of individual losses."""
        # If individual_losses is a 1D tensor of shape (num_tasks,), perform element-wise multiplication
        if isinstance(individual_losses, torch.Tensor) and individual_losses.dim() == 1 and individual_losses.size(0) == self.num_tasks:
            # Element-wise multiplication with task_weights
            weighted_loss_tensor = self.task_weights * individual_losses
            # Return as a list of scalar tensors for compatibility with other parts expecting a list
            return [weighted_loss_tensor[i] for i in range(self.num_tasks)]
        else:
            # Fallback for list input (original behavior if needed elsewhere)
            return [self.task_weights[t] * individual_losses[t] for t in range(self.num_tasks)]

    def compute_total_loss(self, weighted_losses):
        """Computes the total loss from weighted losses."""
        return sum(weighted_losses)

    def _calculate_grad_norms(self, weighted_losses_tensor_list):
        """Calculates gradient norms for each task w.r.t. shared parameters."""
        grad_norms = []
        # Iterate through the list of weighted losses (scalars)
        for i in range(self.num_tasks):
            # Compute gradient of the i-th weighted loss w.r.t. shared params
            # retain_graph=True is crucial IF the main model backward happens afterwards without retain_graph.
            # Since we changed the order above (GradNorm first, then main), we might not strictly need it,
            # but it's safer if there are complex interactions or if retain_graph is needed internally by PyTorch.
            # allow_unused=True handles layers not contributing to specific task losses
            # create_graph=False: We don't need higher-order derivatives here for the norm calculation itself.
            grads = torch.autograd.grad(
                weighted_losses_tensor_list[i],
                self.shared_params,
                retain_graph=True,  # Keep graph alive during GradNorm's internal grad calc
                allow_unused=True,
                create_graph=False # Not calculating gradients of gradients here
            )
            squared_norms = []
            for g in grads:
                if g is not None:
                    squared_norm = g.pow(2).sum()
                    squared_norms.append(squared_norm)
            if not squared_norms:
                grad_norm_squared = torch.tensor(0.0, device=self.task_weights.device)
            else:
                grad_norm_squared = torch.stack(squared_norms).sum()
            grad_norm = grad_norm_squared.sqrt()
            grad_norms.append(grad_norm)
        return grad_norms

    def _compute_gradnorm_loss(self, grad_norms, current_losses_tensor):
        """Computes the GradNorm loss component."""
        if self.initial_losses is None:
            # If initial losses are not set, return a tensor that maintains graph connection but contributes 0 loss
            if len(grad_norms) > 0:
                 return sum(g * 0 for g in grad_norms)
            else:
                 return torch.tensor(0.0, requires_grad=True)

        # current_losses_tensor is expected to be a 1D tensor of shape (num_tasks,)

        # Calculate rates (how much each task has decreased relative to its initial loss)
        rates = current_losses_tensor / self.initial_losses
        
        # Calculate target inverse rates
        target_norms = self.initial_losses.mean() / rates # This is differentiable

        # GradNorm loss: difference between target norm and actual norm
        gradnorm_loss_components = [torch.abs(target_norms[i] - grad_norms[i]) for i in range(self.num_tasks)] # torch.abs
        gradnorm_loss_val = sum(gradnorm_loss_components) # This sum is differentiable
        return gradnorm_loss_val

    def update_weights_if_needed(self, individual_losses_tensor):
        """
        Updates task weights if the update frequency is reached.
        Expects individual_losses_tensor to be a 1D tensor of shape (num_tasks,).
        """
        self.step_counter += 1
        if self.step_counter % self.update_frequency == 0:

            # Verify input type and shape
            if not (isinstance(individual_losses_tensor, torch.Tensor) and 
                    individual_losses_tensor.dim() == 1 and 
                    individual_losses_tensor.size(0) == self.num_tasks):
                raise TypeError(f"Expected individual_losses_tensor to be a 1D tensor of shape ({self.num_tasks},), got {type(individual_losses_tensor)} with shape {individual_losses_tensor.shape}")

            # Calculate weighted losses for gradient norm calculation
            # Pass the 1D tensor to compute_weighted_losses which handles tensor input
            weighted_losses_for_update = self.compute_weighted_losses(individual_losses_tensor)
            
            # Calculate gradient norms based on these weighted losses
            grad_norms_for_update = self._calculate_grad_norms(weighted_losses_for_update)

            # Calculate the GradNorm loss using the current losses tensor and calculated norms
            gradnorm_loss_for_optim = self._compute_gradnorm_loss(grad_norms_for_update, individual_losses_tensor)

            # Update weights using the GradNorm loss
            self.optimizer.zero_grad()
            gradnorm_loss_for_optim.backward(retain_graph=True)
            self.optimizer.step()

            # Apply non-negativity constraint
            with torch.no_grad():
                self.task_weights.clamp_(min=0.0)

            if random.random() < 0.001:
                print(f"[GradNorm Update] Weights: {self.task_weights.tolist()}, "
                      f"GradNorm Loss: {gradnorm_loss_for_optim.item():.6f}, "
                      f"Gradient Norms: {[g.item() for g in grad_norms_for_update]}")
            return True # Indicate an update happened
        return False # Indicate no update happened

    def get_current_weights(self):
        """Returns the current task weights."""
        return self.task_weights.clone().detach().cpu().tolist()

    def get_task_weights(self):
        """Alias for getting task weights, matching your print statement."""
        return self.get_current_weights()