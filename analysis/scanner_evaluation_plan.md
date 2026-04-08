Your goal is to update the scanner_performance.ipynb to be more flexible for analyzing scanners data in comparison with the other sources of information (e.g., human data) and with comparing scanners against one another.

There are two contexts to keep in mind for this script: either a single scan file might be passed in, or multiple files may be passed in. The analysis of a single file is the primary objective, but the multi-file option should be a natural extension of this.

** Analyzing a single file **
I want to create an analysis dataframe that is
- 1 row per transcript
- Contains all the metadata for that transcript (model, type, etc) (comes from preprocessed_eval_runs.csv)
- Contains the human labels for that transcript, if present (comes from preprocessed_human_labels.csv)
- Contains the scanner scores for each scanner run against that transcript (e.g., answer_format_grade)
Will want to be able to split things apart by:
- Scanner that was run (eg, answer_matching or ground_truth)
- Dataset that was scanned (eg mini, synth, etc)
- Should also have eval label, though this is always the same in this case.
And I want plots looking at
- Grades for each of data subsets (e.g., default, synth/T5-contamination)
    - These should be stacked bars for each grade level
    - Should be a plot like this for each scanner, and for the human grades
- Violation rates (grade >= 2, this threshold should be optional so it can be changed later) for each data subset by scanner and human grader
- Scanner accuracy, sensitivity, and specificity, using the violation rate calculated above
    - There should be options in the block to specify what is considered the desired target for each data subset.
    - This should be able to use the human data if present, with an option to specify which set of human data is used for which scanner (e.g., the T5 scanners can be paired up with the T5 human data, but not the oh1 human data)
    - It should also be possible to simply label the target as uniform across a dataset, for instance if the synth/t5-contamination should be considered as having a 100% violation rate

** Analyzing multiple files ** 
The main difference here is that there will be independent scanner runs (identified by the scanner_source) which may have the same scanner labels and transcripts. 
-This means the above dataframe will become 1 row per transcript X scanner_source
-The above plots should be adapted to include side by side comparisons for the different scanner runs if run on the same data subsets.
