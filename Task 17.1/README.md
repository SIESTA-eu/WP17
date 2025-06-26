<div align="justify">This subfolder contains the synthetic dataset in the domain of cyberincidents report generated for the purpose of text anonymization experiments and cyberincidents report classification. The instances have been pre-classified according to the class of incidents and a finer subclasses based on specifics.


The synthetic dataset contains fictitious cyber incident prose-based reports, that have been enriched with fictitious personal data. These cyber incidents are labelled using an enhanced version of the INCIBE cyber incident taxonomy. The cyber incidents are in a primarily prose format and mimic incident reports, such as from newspapers, as well as first-person reports, like what may be reported in a cyber incident portal.
The development of effective cybersecurity machine learning models requires large volumes of realistic incident data. However, real cyber incident data contains sensitive information that cannot be shared for research purposes; creating a significant data scarcity problem in cybersecurity AI research. Therefore, this experimentation presents a dataset which contains realistic details across diverse cyber incident classes, while maintaining coherence. This research also includes varying tones (formal and informal) and perspectives (first person to third person), since cyber incident reports may come from a variety of sources. 
The fictitious nature of this dataset makes it invaluable for scenarios where real incident data would pose privacy, legal, or security risks:
- Security Research and Analysis: Developing automated incident detection systems in SIEM platforms
- Training and Education: Educating cybersecurity practitioners and training machine learning models
- Compliance Testing: Workflow testing in regulatory sandboxes
- Academic Research: Benchmarking, validation of cyber incident taxonomies and privacy-preserving research
- Prototyping: Testing incident analysis techniques without privacy concerns



# Data Description
## General Statistics
Size: 10000 records
Average word count per incident: 250 words
Layout: Contains 3 fields, namely TEXT, CLASS and SUB-CLASS
File format: CSV

## Class and Subclass Distribution 
**Availability (2,444 incidents):**
1. Denial of Service: 1,568
2. Unintended Interruption: 148
3. Misconfiguration: 157
4. Sabotage: 189
5. Distributed Denial of Service: 187
6. Outage: 185

**Abusive Content (827 incidents):**
1. Exploitation/Sexual Harrassment/Violent Content: 242
2. Hate Crime: 238 
4. Spam: 171

**Information Gathering (613 incidents):**
1. Social Engineering: 233
2. Sniffing: 205
3. Scanning: 198

**Information Content Security (1,170 incidents):**
1. Unauthorised Access to Information: 403
3. Unauthorised Modification of Information: 691
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
1. Login Attempts: 298

**Vulnerable (490 incidents):**
1. Weak Cryptography: 217 
2. Vulnerable System: 273 

**Other (150 incidents):**
1. Other: 150 

### TO DO:
We will completely annotate all the personally identifiable information (PII) in the samples</div> 
