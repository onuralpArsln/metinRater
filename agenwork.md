 python -m pipenv run python run_all_targets.py

First use extractor.py on all html files on target folder.
This will create successful.txt and unsuccessful.txt files.
succesful.txt contains the top 20 product names from html files.
unsuccessful.txt contains the bottom 20 product names from html files.
Also compy them as successful.txt->grup1.txt and unsuccessful.txt-> group2.txt for anonymous llm analysis
run all test for each line in test_texts.txt (each line is handwritten by user)
tests should compare whole succesful targets at once. 4 target means 4*20 = 80 lines of succesful header. so apply test for each line from test_texts.txt against all succesful and unsuccessful headers at once. 


analyze the test results as csv  as in example 
Target,Test1_Score,Test2_Score,Test3_Score,Test4_Score,Test5_Score,Test6_Score,Test8_Score,Test7_Score,Final_Verdict,Final_Confidence
beldestyas-target.html,0.385 (S),0.282 (U),56.5 (U),0.428 (U),52.4 (S),54.3 (U),28.2 (U),53.8 (U),UNSUCCESSFUL,53.8
boydesortyas.html,0.332 (U),0.295 (U),59.7 (U),0.231 (S),54.9 (U),59.4 (U),34.5 (U),56.1 (U),UNSUCCESSFUL,56.1
boyfityas.html,0.390 (S),0.276 (S),53.3 (S),0.380 (S),50.1 (U),50.6 (U),16.4 (S),51.1 (S),SUCCESSFUL,51.1

summarize csv as in example 
target-Final_Verdict-Final_Confidence-unsucces_votes-succes_votes
beldestyas-target.html-UNSUCCESSFUL-0.332-6-1
boydesortyas.html-SUCCESSFUL-0.390-6-1
boyfityas.html-UNSUCCESSFUL-0.439-2-5



