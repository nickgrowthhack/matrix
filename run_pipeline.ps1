# Build the mold end-to-end, then generate. Run from anywhere.
$ErrorActionPreference = "Stop"
$py  = "C:\matrix\.venv\Scripts\python.exe"
$rs  = "C:\Program Files\R\R-4.6.0\bin\Rscript.exe"
$env:R_LIBS_USER = "C:\Users\nickg\AppData\Local\R\win-library\4.6"

Write-Host "== S1 binarize + grid + segment ==" -ForegroundColor Cyan
& $py C:\matrix\python\s1_binarize_segment.py
Write-Host "== S2 classify ==" -ForegroundColor Cyan
& $py C:\matrix\python\s2_classify.py
Write-Host "== S3 contours ==" -ForegroundColor Cyan
& $py C:\matrix\python\s3_contours.py
Write-Host "== S3b align ==" -ForegroundColor Cyan
& $py C:\matrix\python\s3b_align.py
Write-Host "== S4 Momocs EFA + PCA shape model (R) ==" -ForegroundColor Cyan
& $rs C:\matrix\R\s4_shape_model.R
Write-Host "== S5 layout / composition model ==" -ForegroundColor Cyan
& $py C:\matrix\python\s5_layout_model.py
Write-Host "== generate ==" -ForegroundColor Cyan
& $py C:\matrix\python\generate.py
Write-Host "== S7 validate ==" -ForegroundColor Cyan
& $py C:\matrix\python\s7_validate.py
Write-Host "DONE. Mold in data/model/, image in output/, reports in reports/." -ForegroundColor Green
Write-Host "Generate variations:  & '$py' C:\matrix\python\generate.py <seed>"
