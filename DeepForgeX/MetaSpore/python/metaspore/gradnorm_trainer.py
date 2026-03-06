import asyncio
from .distributed_trainer import DistributedTrainer

class GradNormTrainer(DistributedTrainer):
    """
    A trainer class that extends DistributedTrainer to incorporate GradNorm logic.
    It handles the GradNorm weight updates followed by the standard distributed training step.
    """

    def __init__(self, gradnorm_manager, *args, **kwargs):
        """
        Initializes the GradNormTrainer.

        Args:
            gradnorm_manager: An instance of GradNormManager to handle weight updates.
            *args: Arguments to be passed to the parent DistributedTrainer constructor.
            **kwargs: Keyword arguments to be passed to the parent DistributedTrainer constructor.
        """
        # Initialize the parent DistributedTrainer
        super().__init__(*args, **kwargs)
        self.gradnorm_manager = gradnorm_manager

    def train(self, individual_losses_for_gradnorm, aux_reg_loss_value):
        """
        Performs a training step incorporating GradNorm.

        Args:
            individual_losses_for_gradnorm: List of individual task loss tensors used for GradNorm weight updates
                                            and subsequently for calculating the total loss for main update.
            aux_reg_loss_value: The auxiliary regularization loss tensor (gate_l2, expert_dissim, etc.).
                                This is added to the weighted task losses.
        """
        if not self.model.training:
            message = "model is in evaluation mode, cannot train it; "
            message += "call the 'train' method to set it in training mode explicitly"
            raise RuntimeError(message)

        # --- GradNorm-Specific Logic ---
        # 1. Update GradNorm weights (W_i) based on individual losses from the *current* mini-batch state.
        # This modifies self.gradnorm_manager.alphas inplace.
        gradnorm_updated = self.gradnorm_manager.update_weights_if_needed(individual_losses_for_gradnorm)

        # --- Recompute Total Loss AFTER GradNorm weights might have changed ---
        # This ensures the total_loss used for main model update uses the LATEST alphas.
        # 2a. Compute weighted losses using the possibly updated GradNorm weights
        weighted_losses_for_main_update = self.gradnorm_manager.compute_weighted_losses(
            individual_losses_for_gradnorm # Use the same tensors passed in
        )

        # 2b. Compute the *primary* prediction loss as the sum of these NEW weighted losses
        primary_prediction_loss = self.gradnorm_manager.compute_total_loss(weighted_losses_for_main_update)

        # 2c. Calculate the new total loss for the main model parameter update
        total_loss_for_main_update = primary_prediction_loss + aux_reg_loss_value

        # --- Standard Distributed Training Logic ---
        # 3. Zero gradients for main model parameters
        self.model._zero_grad()

        # 4. Backward pass for the main model parameters using the RECOMPUTED total_loss.
        # This loss is now guaranteed to be based on the current alphas.
        total_loss_for_main_update.backward(retain_graph=False) # Still usually False after GradNorm step

        # 5. Push gradients (distributed part)
        asyncio.run(self.model._push_tensors(skip_no_grad=self.skip_no_grad))

        # Optional: Log GradNorm weights if they were updated during this step
        if gradnorm_updated:
             # Example logging - adjust based on your logging mechanism
             # print(f"GradNorm weights updated: {self.gradnorm_manager.get_task_weights()}")
             pass # Or integrate with your existing logging framework