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

from .mmoe.mmoe_net import MMoE
from .mmoe.mmoe_mdl_net import MdlMMoEModel
from .mmoe.mmoe_mtl_net import MtlMMoEModel
from .mmoe.mmoe_agent import MMoEAgent
from .mmoe.scene_mmoe_mtl import SceneAwareMMoE
from .mmoe.home_net import HoME

from .esmm.esmm_net import ESMM
from .esmm.esmm_agent import ESMMAgent

from .shared_bottom_net import MtlSharedBottomModel
from .ple_md_net import MultiLayerPLEMD
from .pepnet_net import PEPNet
from .pepnet2_net import PEPNet2
from .steam import STEMNet
from .m2m_net import M2MModel