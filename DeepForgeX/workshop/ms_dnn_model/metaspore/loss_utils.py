#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch

def nansum(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x).sum()

def log_loss(yhat, y, eps=1e-12):
    # Clamp yhat to [eps, 1 - eps] for numerical stability
    yhat = torch.clamp(yhat, eps, 1 - eps)
    loss = -(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))
    return nansum(loss)

class LossUtils:
    @staticmethod
    def log_loss(yhat, y, eps=1e-12):
        return log_loss(yhat, y, eps)
    
    @staticmethod
    def nansum(x):
        return nansum(x)