import os,sys
import shutil
import re
from typing import Union
import pandas as pd
from tqdm import tqdm

from src.tools import file_tools, path_tools
from src.data.data_tool import get_data_path
# from src.label_matrix.basemodel_rule ...
from src.label_matrix.DetailedLabel import DetailedLabel

from src.label_matrix.struct_rule.main_struct import StructRuleModule
from src.label_matrix.semantic_rule.main_semantic import SemanticRuleModule
from src.label_matrix.semantic_rule.wordlist.chatGPT_semantic import get_domain_from_chatGPT
from src.label_matrix.semantic_rule import semantic_similarity

from src.label_matrix.basemodel_rule.llmquery import LogAnnotationPromptGenerator

# sim_th in drain is 0.7, get total total XXXX template

def isBGLError(s):
    if s.startwith('-'):
        return 0
    else:
        return 1

class LabelAnnotationBGL:
    SEMANTIC_RM = SemanticRuleModule()
    def __init__(self,log_template) -> None:
        self.log_template = log_template
        self.groundtruth_label:int = 1 if isBGLError() else 0
        self.action,self.status,self.domain_knowledge = LableAnnotation.SEMANTIC_RM.generate_semantic_label(log_template)
        #* basemodel rule: only write the prompts
        self.llmprompt = None
        self.log_template_llm = None
        self.llm_ref_label_list = []
        self.llm_final_numeric_label = None

def main_process():
    all_template = ...
    res:list[LabelAnnotationBGL] = []


class LableAnnotation:
    STRUCT_RM = StructRuleModule()
    SEMANTIC_RM = SemanticRuleModule()
    def __init__(self,log_template:str,iid) -> None:
        self.iid = iid
        self.log_template = log_template
        
        self.is_test = False
        self.is_have_model_annotated_result = False
        
        self.event_level_label_type = None
        self.suggestion_label_type = None
        
        
        #* struct rule
        self.event_level,self._event_identifier,self._suggestion,self.heuristics_source = LableAnnotation.STRUCT_RM.generate_struct_label(log_template)
        
        if self._event_identifier == None and self._suggestion == None:
            self.fullsuggestion = ""
        elif self._event_identifier == None and self._suggestion != None:
            self.fullsuggestion = f"suggestion: {self._suggestion}"
        elif self._event_identifier != None and self._suggestion == None:
            self.fullsuggestion = f"event type: {self._event_identifier}"
        else:
            self.fullsuggestion = f"event type: {self._event_identifier}; suggestion: {self._suggestion}"
        
        self.HeruisticLabel = DetailedLabel(
            annotation_level='struct',
            annotation_method_name=self.heuristics_source,
            label_content_dict={
                'event_level':int(self.event_level) if self.event_level is not None else -1,
                'suggestion': self.fullsuggestion
            },
            reference_content_list=[]
        )
        
        
        #* semantic rule
        self.action,self.status,self.domain_knowledge = LableAnnotation.SEMANTIC_RM.generate_semantic_label(log_template)
        self.SemanticLabel = DetailedLabel(         #* default to be empty label
            annotation_level='semantic',
            annotation_method_name= "manual_semantic",
            label_content_dict = {
                'event_level': -1,
                'suggestion': None
            },
            reference_content_list=[]
        )
        self.semantic_boot_type = "original"
        
        #* basemodel rule: only write the prompts
        self.llmprompt = None
        self.log_template_llm = None
        self.llm_ref_label_list = []
        self.llm_final_numeric_label = None
    
    def isSemanticBoost(self):
        return self.semantic_boot_type != "original"
    def isLLMBoost(self):
        return self.llmprompt != None
    
    def getSemantic(self)->dict:
        return {
            "action":self.action,
            "status":self.status,
            "domain_knowledge":self.domain_knowledge
        }
    
    def __jsonencode__(self):
        return {
            "log_template":self.log_template,
            
            #* STRUCT_RULE
            "event_level": self.event_level,
            "event_identifier": self._event_identifier,
            "suggestion": self._suggestion,
            "heuristics_source": self.heuristics_source,
            
            #* SEMANTIC_RULE
            "semantic_action": self.action,
            "semantic_status": self.status,
            "semantic_domain": self.domain_knowledge,
            
            #* BASEMODEL_RULE
            "model_prompt": ...,
        }
    
    def __str__(self) -> str:
        return "\n".join([
            f"TEMPLATE: {self.log_template}",
            f"SRTUCT_RULE:",
            f"|--------event_level: {self.event_level}",
            f"|---event_identifier: {self._event_identifier}",
            f"|---------suggestion: {self._suggestion}",
            f"|--heuristics_source: {self.heuristics_source}",
            f"SEMANTIC_RULE:",
            f"|--action: {self.action}",
            f"|--status: {self.status}",
            f"|--domain: {self.domain_knowledge}"
        ])

#* main
def main_annoataion_entrance():
    # file_tools.tmp_clear()
    
    
    #*========== 1. load data, and apply 
    #* - struct rule: get groundtruth label
    #* - semantic rule: get semantic information

    all_template = [] #1542 in total
    
    all_log_template_datafolder = get_data_path("raw_logtemplate_dir")
    all_template = file_tools.txtline_load(os.path.join(all_log_template_datafolder,'gather.txt'))
    
    # for file in os.listdir(all_log_template_datafolder):
    #     each_file_path = os.path.join(all_log_template_datafolder,file)
    #     all_template.extend(file_tools.txtline_load(each_file_path))
    
    res:list[LableAnnotation] = []
    idx = 0
    for each_template in tqdm(all_template):
        each_L = LableAnnotation(each_template,idx)
        idx+=1
        res.append(each_L)
        
    print(LableAnnotation.STRUCT_RM.statistic_scaned_success)
    print(f"TOTAL:{LableAnnotation.STRUCT_RM.statistic_scaned_total}")


    #* add all numeric labels
    all_labels = list(set([r.HeruisticLabel.label_content_dict['suggestion'] for r in res]))
    
    from src.human_labeler.llm_labeler_in import HumanLabelPersist
    p_label = HumanLabelPersist("suggestion_all")  # * use persist label
    p_label.extend_label(["0", "1", "2"])
    p_label.extend_label_persist(all_labels)


    
    for r in res:
        r.HeruisticLabel.set_final_label_by_persistant_label(p_label)
    
    
    print("IN SEMANTIC BOOST .......")
    print(f"sentiment total: {LableAnnotation.SEMANTIC_RM.statistic_status}")
       
    #*========== 2. semantic rule base boost: boost the semantic information by using semantic similarity    

    # print(LableAnnotation.SEMANTIC_RM.statistic_status)           # {'POSITIVE': 51, 'NEGATIVE': 898, 'UNKNOWN': 593}
    
    #* classify all negative by level
    
    #* ~~~~~~~~~~~~~~~~~~~~~~~~
    def get_level_dict_iid_by_semantic_status(status:str):
        level_dict_iid = {'0':[],'12':[],'3':[]}  # 0 -> 没有; 1-> 只有event_level; 2-> 只有suggestion; 3-> 两个都有
    
        for i in res:
            if i.status[1]==status:
                l = i.HeruisticLabel.label_level()
                if l in [1,2]:
                    level_dict_iid['12'].append(i.iid)
                else:
                    level_dict_iid[str(l)].append(i.iid)
        _0123_to_meaning = {'0':'no label','12':'half label','3':'full label'}
        _level_dict_len = {_0123_to_meaning[k]:len(v) for k,v in level_dict_iid.items()}
        print(f"-- Status: level of {status} | {_level_dict_len}")
        return level_dict_iid
    positive_level_dict_iid = get_level_dict_iid_by_semantic_status('POSITIVE')
    negative_level_dict_iid = get_level_dict_iid_by_semantic_status('NEGATIVE')
    unknown_level_dict_iid = get_level_dict_iid_by_semantic_status('UNKNOWN')
    #* ~~~~~~~~~~~~~~~~~~~~~~~ 
    
    print("==="*10)
    
    

    
    def semantic_boost(res,source_level_dict,reference_level_dict,keywd_semantic_prefix:str):
        """

        Args:
            source_level_dict (_type_): {"0":list(id),"12":list(id),"3":list(id)}
            reference_level_dict (_type_): 同上

        Returns:
        """
        #* we only use negative to boost
        #* boost 1,2 to 3
        num_12_to_3 = 0
        
        for iid in source_level_dict['12']:        
            matchedlist = []
            for exist_iid in reference_level_dict['3']:
                if semantic_similarity.cal_semenatic_similarity(res[iid].getSemantic(),res[exist_iid].getSemantic()):
                    matchedlist.append(exist_iid)
            
            if len(matchedlist) != 0:
                num_12_to_3 += 1
                res[iid].semantic_boot_type = f"{keywd_semantic_prefix}_12_to_3"
                res[iid].SemanticLabel.change_label_by_vote_aggregation([res[_i].HeruisticLabel for _i in matchedlist])
                res[iid].SemanticLabel.set_final_label_by_persistant_label(p_label)
        
        #* boost 0 to 1,2 or to 3
        num_0_to_12 = 0
        num_0_to_3 = 0

        for iid in source_level_dict['0']:
            matchedlist0to12 = []
            for exist12 in reference_level_dict['12']:
                if semantic_similarity.cal_semenatic_similarity(res[iid].getSemantic(),res[exist12].getSemantic()):
                    matchedlist0to12.append(exist12)
        
            matchedlist0to3 = []
            for exist3 in reference_level_dict['3']:
                if semantic_similarity.cal_semenatic_similarity(res[iid].getSemantic(),res[exist3].getSemantic()):
                    matchedlist0to3.append(exist3)
            
            if len(matchedlist0to3) != 0:
                num_0_to_3 += 1
                res[iid].semantic_boot_type = f"{keywd_semantic_prefix}_0_to_3"
                res[iid].SemanticLabel.change_label_by_vote_aggregation([res[_i].HeruisticLabel for _i in matchedlist0to3])
                res[iid].SemanticLabel.set_final_label_by_persistant_label(p_label)
            elif len(matchedlist0to12) != 0:
                num_0_to_12 += 1
                res[iid].semantic_boot_type = f"{keywd_semantic_prefix}_0_to_12"
                res[iid].SemanticLabel.change_label_by_vote_aggregation([res[_i].HeruisticLabel for _i in matchedlist0to12])
                res[iid].SemanticLabel.set_final_label_by_persistant_label(p_label)
            else:
                ...
        # [r.iid for r in res if r.semantic_boot_type=='0_to_3']

        #* Statistics

        sum_sem_boost = num_0_to_12+num_0_to_3+num_12_to_3
        print(f"Semantic boost {keywd_semantic_prefix}  status: {sum_sem_boost}/{len(res)}, boost coverage :{100*sum_sem_boost/len(res)}%")

        print(f"-- 0to12:{num_0_to_12}")
        print(f"-- 0to3:{num_0_to_3}")
        print(f"-- 12to3:{num_12_to_3}")
        
        print(f"-- level 3:{len(source_level_dict['3'])} -> {len(source_level_dict['3'])+num_12_to_3+num_0_to_3}")
        print(f"-- level 12:{len(source_level_dict['12'])} -> {len(source_level_dict['12'])+num_0_to_12-num_12_to_3}")
        print(f"-- level 0:{len(source_level_dict['0'])} -> {len(source_level_dict['0'])-num_0_to_3-num_0_to_12}")
    
    #* semantic boost: negative
    semantic_boost(res,negative_level_dict_iid,negative_level_dict_iid,"NEGATIVE")
    
    #* semantic boost: unknown
    unk_plus_neg = {i:negative_level_dict_iid[i]+unknown_level_dict_iid[i] for i in ['0','12','3']}
    semantic_boost(res,unknown_level_dict_iid,unk_plus_neg,"UNKNOWN")
    
    # file_tools.tmp_pickle_save(res,"before_llm.pkl")
    # res = file_tools.tmp_pickle_load('before_llm.pkl')
    #* ========== 3. iterate and generate basemodel label
    
    #* replace \ in llm results
    for r in res:
        r.log_template_llm = r.log_template.replace("\\n","").replace("\\","")
    
    #* reference
    llm_reference_json_path = get_data_path("llm_reference.json")
    def generate_llm_reference():
        ret_dict = {}
        for r in res:
            ret_dict[r.log_template_llm] = {"suggestion":r.HeruisticLabel.get_final_label()}
        file_tools.json_save(ret_dict,llm_reference_json_path)
    generate_llm_reference()
    
    #* generate all prompts
    pg = LogAnnotationPromptGenerator("src/data/llm_reference_all.json")
    
    llmjob_json_dict_list = [] 
    for r in tqdm(res):
        # if r.HeruisticLabel.label_level() == 0 or (r.HeruisticLabel.label_level() == 1 and r.HeruisticLabel.label_content_dict['event_level'] == '0'):
        if r.HeruisticLabel.label_level() == 0 or (r.HeruisticLabel.label_level() == 1):
            r.llmprompt,r.llm_ref_label_list =  pg.get_prompt(r.log_template_llm)
            llmjob_json_dict_list.append({
                "iid":r.iid,
                "log_template":r.log_template_llm,
                "prompt":r.llmprompt
            })
    file_tools.json_save(llmjob_json_dict_list,get_data_path("llm_job.json"))
    
    # breakpoint()
    
    
    file_tools.tmp_pickle_save(res,"final.pkl")
    res = file_tools.tmp_pickle_load("final.pkl")
    
    #* =========== 3.5 write back llm results
    llm_result_path = get_data_path("llm_jobfinish-qianwen7B.json")
    llm_result = file_tools.json_load(llm_result_path)
    
    
    
    def aggregate_llm_result(llm_res_list:list[str],llm_ref_label_list:list)->int:
        numlist = []
        for s in llm_res_list:
            r = re.search(r"[\d]+",s)
            num = -1 if r == None else int(r.group())
            if num not in llm_ref_label_list:
                num = -1
            numlist.append(num)
    
    
        # majority_volt llm result
        dedupe = list(set(numlist))
        if len(dedupe) == 0:
            return -1
        if len(dedupe) == 1:
            return dedupe[0]
        if len(dedupe) == 2:
            count1 = numlist.count(dedupe[0])
            count2 = numlist.count(dedupe[1])
            return dedupe[0] if count1 > count2 else dedupe[1]
        if len(dedupe) == 3:
            return -1
        

              
    
    for i in llm_result:
        _iid = i['iid']
        res[_iid].llm_final_numeric_label = aggregate_llm_result(
            [i['llm_label_1'],i['llm_label_2'],i['llm_label_3']], res[_iid].llm_ref_label_list)

    
    
    
    #* ========================  export id
    
    pd_list = []
    for r in res:
        heruistic_label = int(r.HeruisticLabel.get_final_label())
        semantic_label = int(r.SemanticLabel.get_final_label())
        llm_label = r.llm_final_numeric_label if r.llm_final_numeric_label != None else -1
        
        final_semlabel = semantic_label if semantic_label != -1 else heruistic_label
        final_llmlabel = llm_label if llm_label != -1 else heruistic_label

        pd_list.append({
            'iid':r.iid,
            'log':r.log_template,
            'heruistic_label':heruistic_label,
            'semantic_label':final_semlabel,
            'llm_label':final_llmlabel
        })
    retpd = pd.DataFrame(pd_list)
    retpd.to_csv(get_data_path("final.csv"),index=False)
    print('finish to final.csv')
        
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print(123)

    
    
    


if __name__ == "__main__":
    main_annoataion_entrance()
    # import csv
    # # Define CSV filename and header row
    # filename = "annotation.csv"
    # header = res[0].keys()

    # # Open CSV file in write mode
    # with open(filename, "w", newline="") as file:

    #     # Create CSV writer object
    #     writer = csv.writer(file)

    #     # Write header row to CSV file
    #     writer.writerow(header)
    #     for i in res:
    #     # Write data rows to CSV file
    #         writer.writerow(i.values())
    
    #*========================================================================================================================

    #*========================================================================================================================
    

    # def count_domain_words(obj_list):
    #     d_l = []
        
    #     for obj in obj_list:
    #         domain = obj.getSemantic()['domain_knowledge']
    #         if domain!=[]:
    #             wordlist,_level = zip(*domain)
    #         d_l.extend([dword for dword in wordlist])
    #     res = {}
    #     for w in d_l:
    #         if w not in res.keys():
    #             res[w] = 1
    #         else:
    #             res[w] +=1
                
    #     return res
            
        
    # obj_list = file_tools.pickle_load('tmp.pkl')
    # r = count_domain_words(obj_list)
    # filtered_r = {k:v for k,v in r.items() if v > 1}
    # sorted_r = sorted(filtered_r.items(),key=lambda x:x[1],reverse=True)
    # print(sorted_r)
        