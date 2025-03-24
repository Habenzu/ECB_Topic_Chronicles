# ECB_Topic_Chronicles
 Repo for the Interdisciplinary Project together with the WU legal tech center in the Master Data Science at the TU Vienna. 

conda environment: dsa

**Data:** 
- "data/all_ECB_speeches.csv": https://www.ecb.europa.eu/press/key/html/downloads.en.html
  - `date`
  - `speakers`
  - `title`
  - `subtitle`
  - `contents` 
- "data/export_datamart.csv": https://www.kaggle.com/datasets/robertolofaro/ecb-speeches-1997-to-20191122-frequencies-dm/data
  - `speech_id`
  - `when_speech`
  - `who`
  - `what_title`
  - `what_frequencies`
  - `what_language`
  - `what_weblink`
  - `what_type`:
    - S: speech
    - I: interview
    - P: press_conference
    - B: blog_post
    - E: podcast
- "data/speeches/parsed.jsonl": Took url from `export_datamart.csv`, downloaded and parsed the html: 
  - `content`
  - `related_topics`
  - `speech_id`
  - `date`
  - `type_long`




## Approach 

1. Download or scrape the data from ECB website
   - **Information to retrieve:** 
     - type (str): press release or blog article
     - date (datetime): date of the publication (DD.MM.YYYY)
     - url (str): the url to the publication, especially important when scraping instead of structured download 
     - authors (array of str): list of authors of the text, for press releases author is "ECB"
     - title (str): title of the publication
     - text (str): full text of the publication, bullet point lists are sperated by comma, headers are just normal text, images are ignored
     - 
   - Put data into a suitable databse (postgres?)
2. Modelling Pipeline: 
   1. 