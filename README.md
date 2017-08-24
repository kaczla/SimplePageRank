# SimplePageRank

**SimplePageRank** is simple project whose purpose is compute PageRank base on Polish People indexed in [Polish Wikipedia](https://pl.wikipedia.org).

---
**RUN:**
1) Download [Polish Wikipedia dump](https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles.xml.bz2) (~1.5GB or more)
2) Unpack archive (after unpack file weighs ~8GB)
3) Run script: `./run.py`

---

**OPTIONS:**

  **-h --help**          -> this help message
  
  **-n**                 -> how many people print as result (defualt: 10)
  
  **-f --file [name]**   -> Wikipedia file name (default: plwiki-pages-articles.xml)

  **-p --pickle**        -> use lib-pickle to save variables (default: disabled)
  
       + out_pickle_wiki_author.bin -> Wikipedia pages
       
       + out_pickle_matrix.bin      ->  PageRank matrix
       
       + out_pickle_matrix_col.bin  ->  matrix mapper: convert index to name

  **-m --matrix**        -> calculate matrix (ignore saved variable by pickle, if was used **--pickle** ; default: disabled)
  
  **--row=[number]**     -> max row in PageRank matrix (default: 0)
  
  **--percent=[number]** -> max percent row (where max row is equal: percer * numer of people in Wikipedia) in PageRank matrix (default: 0.25)
  if **--row** and **--percent** was added, then is chosen greater number of them
  
  **--power=[number]**   -> how many times power matrix (default: 3)

  **-w --wiki**          -> search in Wikipeda file (ignore saved variable by pickle, if was used **--pickle** ; default: disabled)

  **--search**          -> get extra information: reference page (not people page) to poeple
  
  **-s --save**          -> save all data to file (default: disabled)
  
       + stage_1.xml                          -> Wikipedia pages where was found people
       
       + stage_2.xml                          -> information about people
       
       + out_ranking_back_reference.txt       -> sorted list by back_reference, line format: `name [tab] value`
       
       + out_ranking_forward_reference.txt    -> sorted list by forward_reference, line format: `name [tab] value`
       
       + out_ranking_back_reference_other.txt -> sorted list by back_reference_other, line format: `name [tab] value`
       
       + out_ranking_back_reference_sum.txt   -> sorted list by (back_reference + back_reference_other), line format: `name [tab] value`
       
       + out_matrix_row_first_label.txt       -> label of first PageRank row, elements are separated by new line
       
       + out_matrix_row_first_ref.txt         -> how many reference have each element, elements are separated by new line 
       
       + out_matrix_row_first.txt             -> first PageRank row, elements are separated by new line

  **--gpu**            -> use GPU acceleration (using Nvidia CUDA)
  
  ---
  
  **REQUIREMENT:**
  * lxml
  * numpy
  * cudamat
