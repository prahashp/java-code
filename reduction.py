import os,time,json
import torch
import scipy.io
import pandas as pd
import numpy as np
import yaml
from datetime import datetime,timedelta
# from DB_Update import PostgresDB

scriptdir = os.path.dirname(os.path.abspath(__file__))
with open(f'{scriptdir}/config.yaml','r') as fp:
    config=yaml.load(fp,Loader=yaml.FullLoader)

class Reduction:
    
    def __init__(self):
        pass
    
    def sort_img_final(self,x, y, feature):
        gf = feature[x]
        qf = feature[y]
        score = torch.mm(gf, qf.T)
        sscore = ((score * 100) ** 2) / 100
        sscore = sscore.numpy().astype(np.int32)
        avg_score = np.sum(sscore) / np.dot(len(gf), len(qf))
        return avg_score

    def get_oulier_value(self,store_id,day,df):
        
        store_exist = df[df['store_id'] == store_id]
        outlier = int(store_exist[day].iloc[0]) if len(store_exist) else -1 
        return outlier

    def temp_merging(self,cust_ind, feat_mat, json_data,reduce,logger):
        try:
            if reduce:
                temps = sorted(list({*json_data.values()}))
                temp_ids1 = {temp: [int(ind) - cust_ind for ind, t in json_data.items() if t == temp] for temp in temps}
                input_feats = {}
                for temp in temps:
                    reid_feature = [torch.FloatTensor(feat_mat[i]).reshape([1, 512]) for i in temp_ids1[temp]]
                    reid_feature = torch.cat(reid_feature, 0)
                    input_feats[temp] = reid_feature
                del feat_mat
                Merge_ref = {'mergers': [], 'scores': []}
                for i in range(len(temps)):
                    for j in range(i + 1, len(temps)):
                        sco = self.sort_img_final(temps[i], temps[j], input_feats)
                        Merge_ref['mergers'].append((temps[i], temps[j]))
                        Merge_ref['scores'].append(sco)
                df1 = pd.DataFrame(Merge_ref)
                df1 = df1.sort_values(by=['scores'], ascending=False)
                df1.index = range(len(df1))

                unique=[]
                final_merge=[]
                for idx,row in df1.iterrows():
                    final_merge.append(list(row['mergers']))
                    unique.extend([i for i in row['mergers'] if i < 10000])
                    if len(set(unique)) >= int(reduce):
                        unique_groups=len(self.merge_list(final_merge))
                        if len(set(unique)) - unique_groups  >= int(reduce):
                            break
                return final_merge
            else:
                final_merge=[]
                return final_merge
            
        except Exception as e:
            logger.error(f'Exception in temp_merging ..{e}')

    def merge_list(self,pair_collector):
        out1 = []
        while len(pair_collector) > 0:
            first, *rest = pair_collector
            first = set(first)
            lf = -1
            while len(first) > lf:
                lf = len(first)
                rest2 = []
                for r in rest:
                    if len(first.intersection(set(r))) > 0:
                        first |= set(r)
                    else:
                        rest2.append(r)
                rest = rest2
            out1.append(first)
            pair_collector = rest
        return out1     
            
    def recreate_master_json(self,temp_ids,date,store_id,logger):
        try:
            master = {img: temp for temp in temp_ids.keys() for img in temp_ids[temp]}
            reduction_master_json = dict(sorted(master.items(), key=lambda x: x[0]))
            re_path = f"{config['jsonPath']}/{date}/{store_id}/"
            os.makedirs(re_path, exist_ok=True)
            # with open(f'{re_path}/reduction.json', 'w') as fp:
            #     json.dump(reduction_master_json, fp)
            logger.info(f'reduction json created for {store_id}')
            return reduction_master_json
        except Exception as e:
            logger.error(f'Exception in recreate_master_json : {e}')
            return reduction_master_json
            
    def main(self,date,store_id,records_df,gallery_feature,cust_ind,temp_id_json,logger):
        try:
            if 'zipfile_name' in temp_id_json:
                del temp_id_json['zipfile_name']
                del temp_id_json['zipfile_count']
            
            rev_date=datetime.strftime(datetime.strptime(date,'%d-%m-%Y'),'%Y-%m-%d')
            store_exist = records_df[records_df['store_id'] == store_id]
            reduction_percent = (store_exist['reduction_per'].iloc[0]) / 100 if len(store_exist) else 0  
            before_reduction=len([temp for temp in set(temp_id_json.values()) if temp < 10000])
            reduce = int(reduction_percent * before_reduction)
            logger.info(f'before count without reduction : {before_reduction}')
            final_merge = []
            temp_ids = {temp: [int(ind) for ind, t in temp_id_json.items() if t == temp] for temp in {*temp_id_json.values()}}
            final_merge = self.temp_merging(cust_ind, gallery_feature, temp_id_json,reduce,logger)
            out=self.merge_list(final_merge)
            for merge in out:
                base = max(merge) if 20000 in merge else max(merge) if any([1 for i in merge if i >= 10000]) else min(merge)
                for temp in merge:
                    if temp == base:
                        continue
                    temp_ids[base] += temp_ids[temp]
                    del temp_ids[temp] 

            after_reduction=len([i for i in temp_ids if i < 10000])
            logger.info(f'After reduction count : {after_reduction}')
            reduction_master_json=self.recreate_master_json(temp_ids,date,store_id,logger)
            today=datetime.strptime(date,'%d-%m-%Y')
            weekday_flag=today.weekday() < 5
            day='weekday' if weekday_flag else 'weekend'
            outlier=self.get_oulier_value(store_id,day,records_df)
            audit_flag=1 if after_reduction > outlier else 0
            if reduction_percent==0:
                logger.info(f'store id : {store_id} do not have reduction percent, sending to audit')
                audit_flag=1
                logger.info(f'Audit status : {audit_flag}')
                return audit_flag,temp_id_json
            logger.info(f'threshold value is : {outlier}')
            logger.info(f'Audit status : {audit_flag}')
            # db_obj = PostgresDB(logger)
            # db_obj.execute_query(rev_date, store_id, before_reduction, after_reduction,audit_flag)
            if audit_flag:
                return audit_flag,temp_id_json
            else:
                return audit_flag,reduction_master_json

        except Exception as e:
            logger.error(f'Exception in Reduction : {e}')
            return 1,temp_id_json

        
        