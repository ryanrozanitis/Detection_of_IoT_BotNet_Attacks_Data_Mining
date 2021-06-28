# Detection_of_IoT_BotNet_Attacks_Data_Mining
If you want to play around with this code, by all means go ahead. There is only the one python file, however you also need to download the 6 csv files that contain the Ennio doorbell traffic data. You may also need to make sure that you have the correct packages for the data manipulation to work properly.

This project was an attempt to learn more about SVM and Data Mining while I took a course of the same name at NJIT. I was quite successful in that I obtained a 99.5% correctness rate when attempting to distinguish bening and malign packets traveling through the Ennio Doorbell device using SVM.

This code is written in Python 3.8. I used some sk_learn packages to perform some data manipulation such as StandardScaler and train_test_split. I iterated through the best model using GridSearchCV. I used classification_report and confusion_matrix to display the results in a readable manner.

The 6 csv files are different types of traffic as the names imply. benign_traffic.csv is benign traffic.  tcp.csv is a tcp flood attack. udp.csv is a udp flood attack. scan.csv is port scanning traffic. junk.csv is a denial of service in which a bunch of junk packets are sent to the doorbell device. combo.csv is a mix of all the previous attacks at once. 

SVM itself is intended to be used for one vs the other data analysis, however I set this up in a way that you can try to distinguish more than two types of traffic. I could only obtain an 83.6% correctness rate when attempting to distinguish each type of traffic separately. However, as stated earlier, this is generally not the correct usage of SVM and was only for learning purposes on my part. I wanted to see what else SVM could do.


If you are interested in more technical details, download and read thepdf for my unpublished paper detailing my work and findings.


Traffic data is sourced from:

   -- Creators: Yair Meidan, Michael Bohadana, Yael Mathov, Yisroel Mirsky, Dominik Breitenbacher, Asaf Shabtai and Yuval Elovici
   
   -- Meidan, Bohadana, Mathov, Mirsky, Shabtai: Department of Software and Information Systems Engineering; Ben-Gurion University of the Negev; Beer-Sheva, 8410501; 
   
Israel

   -- Breitenbacher, Elovici: iTrust Centre of Cybersecurity at Singapore University of Technology and Design; 8 Somapah Rd, Singapore 487372
   
   -- Donor: Yair Meidan (yairme@bgu.ac.il)
   
   -- Date: March, 2018 (databases may change over time without name change!)
