## image-translate-pdf ##



This tool allows you to convert a folder of images of non-English language academical physical book into a translated and assembled pdf - which is my best way to tackling the awkwardness of accessing non-English language academical works that can only be obtained in hardback format.



need to add othe rpages types that the VLLM can return the content to be assembled as - such as index, contents, et cetera

## Test ##
image-translate-pdf --input_pdf_path "C:\Users\JoeJo\OneDrive - University of Cambridge\Documents\PhD Documents\Die Buccellarii, Eine Studie  zum militarischen Gefolgschaftswesen in der Spatantike - Oliver Schmitt.pdf" --language German --no-free_tier_llm --type pdf --pdf_path "SchmidtTeest.pdf"


## PyPi Commands ##
1. python -m build
2. python -m twine upload dist/*