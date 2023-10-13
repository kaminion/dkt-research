import os
import pandas as pd



DIVIDE_LISTS = ['FALL', 'SPRING']
BASE_DIR = f"./datasets/CSEDM/"

integrated_fully = pd.DataFrame()
main_fully = pd.DataFrame()
code_fully = pd.DataFrame()

for dir_name in DIVIDE_LISTS:
    dataset_dir = f"{BASE_DIR}{dir_name}"
    
    early = pd.read_csv(f"{dataset_dir}/early.csv").drop_duplicates(subset=['SubjectID', 'ProblemID'])
    late  = pd.read_csv(f"{dataset_dir}/late.csv").drop_duplicates(subset=['SubjectID', 'ProblemID'])
    
    main_table = pd.read_csv(f"{dataset_dir}/MainTable.csv").dropna(subset=["Score"])
    code_state = pd.read_csv(f"{dataset_dir}/CodeStates.csv")
    
    # Code 테이블이랑 main이랑 join, 그리고 early, late 파일 합쳐줌
    main_fully = pd.concat([main_fully, pd.merge(main_table, code_state, on="CodeStateID")])
    concat_fully = pd.concat([early, late])
    
    # 그리고 early late 합친거랑 code join한거랑 다시 join
    main_concat = pd.merge(main_fully, concat_fully, on=["SubjectID", "AssignmentID", "ProblemID"])
    integrated_fully = pd.concat([integrated_fully, main_concat])
    

integrated_fully.to_csv(f"{BASE_DIR}ALL_CSEDM.csv", index=False, encoding='utf-8-sig')