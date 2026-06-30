# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from coreai_opt.config.spec import (
    CompressionComponentFactoryBase,
    CompressionTargetTensor,
)

from .fake_palettize import _FakePalettizeImplBase
from .spec import PalettizationSpec


class _PalettizationComponentFactory(CompressionComponentFactoryBase):
    @classmethod
    def construct(
        cls,
        spec: PalettizationSpec | None,
        target: CompressionTargetTensor = CompressionTargetTensor.WEIGHT,
    ) -> _FakePalettizeImplBase | None:
        pass
