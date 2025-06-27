<div align="justify">This subfolder contains the synthetic dataset in the domain of cyberincidents report generated for the purpose of text anonymization experiments and cyberincidents report classification. The instances have been pre-classified according to the class of incidents and a finer subclasses based on specifics. We created the synthetic dataset using real publicly available cyber incidents descriptions as a foundation. Since these public sources typically contain limited personal information compared to official CERT reports, we enriched the generated reports with fictitious personal data. Additionally, we incorporated insights and structure derived from a small number of real CERT cyber incidents reports to which the ULE research group has access.
Based on INCIBE’s taxonomy,  we identified 10 classes of cyber incidents and several subclasses within them. We modified this taxonomy to incorporate elements from other popular cyber incident taxonomies, such as the ENISA Cyber Incident Taxonomy. These modifications aimed to create a comprehensive and detailed taxonomy, suited for capturing a wide range of cyber incidents.
We used the CECILIA-10C-900 dataset (Delgado et al., 2024) as the basis for the creation of the synthetic dataset. This dataset consists of publicly available cyber incident reports. We reviewed CECILIA-10C-900 and identified at least one cyber incident that would match each class/sub-class pair defined in the modified INCIBE taxonomy.
Each selected cyber incident was then manually enriched with fictitious personal data, either by appending it to the original text, or integrating it into the body of the text. These enriched samples served as “seeds” for the creation of the synthetic dataset. Each class was assigned a number of seed samples proportional to its number of subclasses, with the goal of producing a balanced distribution across subclasses.

The fictitious nature of this dataset makes it invaluable for the following scenarios where real incident data would pose privacy, legal, or security risks:
- Security Research and Analysis: Developing automated incident detection systems in SIEM platforms
- Training and Education: Educating cybersecurity practitioners and training machine learning models
- Compliance Testing: Workflow testing in regulatory sandboxes
- Academic Research: Benchmarking, validation of cyber incident taxonomies and privacy-preserving research
- Prototyping: Testing incident analysis techniques without privacy concerns

# DATA DESCRIPTION
## 1. GENERAL STATISTICS
Size: 10000 records

Average word count per incident: 250 words

Layout: Contains 3 fields, namely TEXT, CLASS and SUB-CLASS

File format: CSV

## 2. TEXT STATISTICS
- Text Length (characters):
  
  Count: 10,000
  
  Mean: 1428.22
  Median: 1429.00
  Std Dev: 144.52
  Min: 180
  Max: 1889
  25th percentile: 1335.00
  75th percentile: 1523.00
  90th percentile: 1612.00
  95th percentile: 1665.00
  99th percentile: 1748.01
  Skewness: -0.3147
  Kurtosis: 1.6793

- Word Count:
  Count: 10,000
  Mean: 196.48
  Median: 197.00
  Std Dev: 20.23
  Min: 26
  Max: 259
  25th percentile: 184.00
  75th percentile: 210.00
  90th percentile: 222.00
  95th percentile: 230.00
  99th percentile: 241.00
  Skewness: -0.3532
  Kurtosis: 1.6217

- Sentence Count:
  Count: 10,000
  Mean: 10.30
  Median: 10.00
  Std Dev: 2.42
  Min: 1
  Max: 31
  25th percentile: 9.00
  75th percentile: 11.00
  90th percentile: 13.00
  95th percentile: 15.00
  99th percentile: 18.00
  Skewness: 1.3854
  Kurtosis: 4.5982

## 3. TOPIC ANALYSIS
Unique topics: 10
Most frequent: Compromiso Informacion (5,028 times)
Least frequent: Intento Intrusion (23 times)

Topics:
   1. Compromiso Informacion: 5,028 (50.28%)
   2. Disponibilidad: 1,432 (14.32%)
   3. Intrusion: 1,294 (12.94%)
   4. Contenido Dañino: 1,226 (12.26%)
   5. Fraude: 379 (3.79%)
   6. Otros: 309 (3.09%)
   7. Vulnerable: 206 (2.06%)
   8. Contenido Abusivo: 80 (0.80%)
   9. Obtencion Informacion: 23 (0.23%)
  10. Intento Intrusion: 23 (0.23%)

## 4. SUBTOPIC ANALYSIS
Unique subtopics: 27
Most frequent: Modificacion No Autorizada (3,356 times)
Least frequent: Servicios Acceso No Deseado (12 times)

Subtopics:
   1. Modificacion No Autorizada: 3,356 (33.56%)
   2. Acceso No Autorizado: 1,683 (16.83%)
   3. DDoS: 1,340 (13.40%)
   4. Compromiso Aplicaciones: 1,169 (11.69%)
   5. Sistema Infectado: 939 (9.39%)
   6. Phishing: 343 (3.43%)
   7. Daños Informaticos: 252 (2.52%)
   8. Sistema Vulnerable: 161 (1.61%)
   9. Distribucion Malware: 103 (1.03%)
  10. DoS: 80 (0.80%)
  11. Servidor C&C: 69 (0.69%)
  12. Revelacion Informacion: 68 (0.68%)
  13. Compromiso Cuenta Privilgios: 68 (0.68%)
  14. Configuracion Malware: 57 (0.57%)
  15. Delito Odio: 57 (0.57%)

## 5. TOPIC-SUBTOPIC PAIRS ANALYSIS
Unique topic-subtopic pairs: 33
Most frequent pair: Compromiso Informacion-Modificacion No Autorizada (3,333 times)
Average occurrences per pair: 303.03
Median occurrences per pair: 34.00
Shannon entropy (pair diversity): 3.0741

Top 20 Topic-Subtopic Pairs:
   1. Compromiso Informacion-Modificacion No Autorizada: 3,333 (33.33%)
   2. Compromiso Informacion-Acceso No Autorizado: 1,683 (16.83%)
   3. Disponibilidad-DDoS: 1,340 (13.40%)
   4. Intrusion-Compromiso Aplicaciones: 1,134 (11.34%)
   5. Contenido Dañino-Sistema Infectado: 939 (9.39%)
   6. Fraude-Phishing: 343 (3.43%)
   7. Otros-Daños Informaticos: 252 (2.52%)
   8. Vulnerable-Sistema Vulnerable: 126 (1.26%)
   9. Contenido Dañino-Distribucion Malware: 103 (1.03%)
  10. Disponibilidad-DoS: 80 (0.80%)
  11. Intrusion-Compromiso Cuenta Privilgios: 68 (0.68%)
  12. Vulnerable-Revelacion Informacion: 68 (0.68%)
  13. Contenido Dañino-Servidor C&C: 57 (0.57%)
  14. Contenido Abusivo-Delito Odio: 57 (0.57%)
  15. Contenido Dañino-Configuracion Malware: 57 (0.57%)
  16. Contenido Dañino-Sistema Vulnerable: 35 (0.35%)
  17. Otros-Otros: 34 (0.34%)
  18. Intrusion-Explotacion Vulnerabilidades: 34 (0.34%)
  19. Intrusion-Vulneracion Credenciales: 23 (0.23%)
  20. Contenido Dañino-Modificacion No Autorizada: 23 (0.23%)

Classification System Overview:
  Topics per dataset: 10
  Subtopics per dataset: 27
  Unique pairs: 33
  Average subtopics per topic: 3.30
  Topic with most subtopics: Contenido Dañino (7 subtopics)


### TO DO:
We will completely annotate all the personally identifiable information (PII) in the samples</div> 
