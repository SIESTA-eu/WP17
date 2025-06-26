<div align="justify">This subfolder contains the synthetic dataset in the domain of cyberincidents report generated for the purpose of text anonymization experiments and cyberincidents report classification. The instances have been pre-classified according to the class of incidents and a finer subclasses based on specifics.


The synthetic dataset contains fictitious cyber incident prose-based reports, that have been enriched with fictitious personal data. These cyber incidents are labelled using an enhanced version of the INCIBE cyber incident taxonomy. The cyber incidents are in a primarily prose format and mimic incident reports, such as from newspapers, as well as first-person reports, like what may be reported in a cyber incident portal.
The development of effective cybersecurity machine learning models requires large volumes of realistic incident data. However, real cyber incident data contains sensitive information that cannot be shared for research purposes; creating a significant data scarcity problem in cybersecurity AI research. Therefore, this experimentation presents a dataset which contains realistic details across diverse cyber incident classes, while maintaining coherence. This research also includes varying tones (formal and informal) and perspectives (first person to third person), since cyber incident reports may come from a variety of sources. 
The fictitious nature of this dataset makes it invaluable for scenarios where real incident data would pose privacy, legal, or security risks:</div>
- Security Research and Analysis: Developing automated incident detection systems in SIEM platforms
- Training and Education: Educating cybersecurity practitioners and training machine learning models
- Compliance Testing: Workflow testing in regulatory sandboxes
- Academic Research: Benchmarking, validation of cyber incident taxonomies and privacy-preserving research
- Prototyping: Testing incident analysis techniques without privacy concerns
  
# Key features of the dataset
- Dynamic Prompting: Introduces variety through varying temperature (creativity) values and utilising conditional instructions
- Multi-Model Generation: Utilises three LLMs (Llama 8B, Llama 70B, Claude AI)  to capture different linguistic patterns and generation styles.
- Diverse Perspectives: Varying formality levels and narrative viewpoints - such as a person recounting an incident in a newspaper article in first-person perspective (less than 5% of the dataset), or a staff member reporting on the technical aspects of a cyber incident report.
- Comprehensive Classification: Dataset is already labelled, with one of ten cyber incident classes, and one sub-classification.
- Technical Detail Integration: Includes realistic technical elements, such as hashes, IP addresses, CVE values, etc.
- Personal Data Integration: Uses the Sharma and Bantan dataset for realistic fictitious personal information, which includes alignment between name and other personal data (email, usernames, etc).


# Data Description
## General Statistics
Size: 10000 records
Average word count per incident: 250 words
Layout: Contains 3 fields, namely TEXT, CLASS and SUB-CLASS
File format: CSV

## Class and Subclass Distribution 
Sub-classes are highlighted, if they were added to the INCIBE Cyber Incident Taxonomy as part of this body of work.

### Adding Phishing to Information Gathering Class
<div align="justify">The GitHub page classifies Phishing, under the Fraud class, specifically when the attacker is masquerading as another entity in order to persuade the user to reveal private credentials.</div>   However, [Anexo 5](https://www.incibe.es/sites/default/files/contenidos/guias/doc/guia_nacional_notificacion_gestion_ciberincidentes.pdf) only mentions Phishing under Information Gathering, and not Fraud. Despite this, there is only one possible sub-category for Phishing, in the  Information Gathering class: Social Engineering. 

Phishing would not belong under Social Engineering, since the guide specifies  (page 15) that social engineering does not involve the use of technology. Following a call to INCIBE on February 24, the tele-operator confirmed that they would place Phishing  (specifically via email) under Information Gathering. Therefore, we added Phishing as a sub-topic under Information Gathering class.

**Availability (2,444 incidents):**
1. Denial of Service: 1,568
2. Unintended Interruption: 148
3. Misconfiguration: 147
4. Sabotage: 159
5. Distributed Denial of Service: 147
6. Outage: 145
7. Disruption of Unknown Cause (or with Insufficient Details): 130

**Abusive Content (827 incidents):**
1. Exploitation/Sexual Harrassment/Violent Content: 242
2. Hate Speech: 238 (The INCIBE taxonomy has a class named Delito de odio (Hate Crime). However,  this subclass includes things like Cyberharrassment,  which are not inherently hate crimes. Therefore, in order to preserve the essence of the sub-class, we have translated the name of the sub-class to “Hate Speech”, instead of “Hate Crime”)
3. Fake News (including Deep Fakes): 176
4. Spam: 171

**Information Gathering (613 incidents):**
1. Social Engineering: 188
2. Sniffing: 155
3. Scanning: 148
4. Phishing: 145

**Information Content Security (1,170 incidents):**
1. Unauthorised Access to Information: 364
2. Leak of Confidential Information: 149
3. Unauthorised Modification of Information: 591
4. Data Loss: 66

**Malicious Code (752 incidents):**
1. Infected System: 429
2. Malware Configuration: 157
3. Malware Distribution: 166

**Fraud (830 incidents):**
1. Masquerade: 165
2. Unauthorised Use of Resources: 170
3. Phishing: 495

**Intrusion (2,403 incidents):**
1. Privileged Account Compromise: 2,037
2. System Compromise: 221
3. Burglary: 145 

**Intrusion Attempts (298 incidents):**
1. Login Attempts: 149 
2. Unknown Attack: 149 

**Vulnerable (490 incidents):**
1. Weak Cryptography: 217 
2. Vulnerable System: 273 

**Other (150 incidents):**
1. APT: 150 

# TO DO:
We will completely annotate all the personally identifiable information (PII) in the samples
