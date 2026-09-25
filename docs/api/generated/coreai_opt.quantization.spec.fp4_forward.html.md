# coreai_opt.quantization.spec.fp4_forward

### coreai_opt.quantization.spec.fp4_forward(tensor)

Round to the nearest FP4 E2M1 value, returned in fp32.

The E2M1 grid is not uniform – `0, 0.5, 1, 1.5, 2, 3, 4, 6` – so a cast cannot do
this. Ties go to the even encoding index, and a magnitude above the grid saturates to
`6.0`. A NaN is passed through unchanged instead.

* **Parameters:**
  **tensor** (*torch.Tensor*) – Values already divided by their scale, in fp32.
* **Returns:**
  The rounded values, in fp32, with any NaN preserved.
* **Return type:**
  torch.Tensor
