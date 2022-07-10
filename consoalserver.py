import nxn_prod as mat
import reduction as red
import time,os,json,glob
import scipy.io
import pandas as pd
from ast import literal_eval
from datetime import datetime, timedelta
import logging
from logging.handlers import TimedRotatingFileHandler
import multiprocessing
from multiprocessing import current_process
import audit_prod_test as audit
import boto3
import yaml
import gc
import gspread
# from DB_Update import RedshiftDB
from oauth2client.service_account import ServiceAccountCredentials
import sys

audit_prep=audit.AuditPrep()
red_obj=red.Reduction()
sqs_resource = boto3.client('sqs')

scriptdir = os.path.dirname(os.path.abspath(__file__))
with open(f'{scriptdir}/config.yaml','r') as fp:
    config=yaml.load(fp,Loader=yaml.FullLoader)

def cluster_within_tempid(df, temp_ids, data, before_count, logger):
    # Search top preds within temp_id
    def recurse_func1(curr_list):
        curr_pool = [lis for lis in curr_list if lis not in pred_pool]
        if not curr_pool:
            return 1
        pred_pool.extend(curr_pool)
        for lis in curr_pool:
            recurse_func1(data[lis]['preds'])

    def recurse_func(curr_list, temp):
        curr_pool = [lis for lis in curr_list if lis not in pred_pool and lis not in sum(cluster, [])]
        if not curr_pool:
            return 1
        pred_pool.extend(curr_pool)
        for lis in curr_pool:
            inp_data = [i for i in data[lis]['preds'] if i in temp_ids[temp]]
            recurse_func(inp_data, temp)

    try:
        final_temps = {}
        for temp in temp_ids.keys():
            if len(temp_ids[temp]) < 200:
                cluster = []
                flat_list = sum(cluster, [])
                for img in temp_ids[temp]:
                    flat_list = sum(cluster, [])
                    if img in flat_list:
                        continue
                    pred_pool = []
                    inp_data = list(set(temp_ids[temp]) & set(data[img]['preds']))
                    recurse_func(inp_data, temp)
                    cluster.append(pred_pool)
                for lis in cluster:
                    if len(lis) > 2:
                        final_temps[temp] = lis
                        break
                else:
                    final_temps[temp] = sum(cluster, [])
            else:
                final_temps[temp] = temp_ids[temp][:5]
        return final_temps
    
    except Exception as e:
        logger.error(f'Exception in cluster_within_tempid : {e}')


def create_mapping_json(temp_id_json,df, final_temps, data,gallery_feature,cust_ind, store_id, logger):
    
    # Matching best preds by removing FA in all temps
    
    def get_single_pred(gallery_feature,cust_ind,thresh,img_index):
        data=mat.single_pred_score(gallery_feature,cust_ind,thresh,img_index)
        return data['preds']
    
    try:
        clust = set(sum([*final_temps.values()],[]))
        mapping_json = {}
        for key in final_temps.keys():
            mapped = []
            one_img_flag=1 if len(final_temps[key])<3 else 0
            all_preds = df[df.index.isin(final_temps[key])]['preds'].to_list()
            curr_temps = df[df.index.isin(clust & set(sum(all_preds, [])))]['temp_id'].to_list()
            curr_temps = list(set(curr_temps))
            if len(curr_temps) > 1:
                mapped.extend([t for t in curr_temps if t not in mapped and t != key])
            if not mapped and one_img_flag:
                for img in final_temps[key]:
                    if store_id.split('-')[0] in config['one_img_clients']:
                        one_thresh = 10
                    else:
                        one_thresh = 40
                    curr_preds=get_single_pred(gallery_feature,cust_ind,one_thresh,img)
                    for pred in curr_preds:
                        if temp_id_json[str(pred)] != key:
                            mapped.append(temp_id_json[str(pred)])
                            break
            mapping_json[key] = list(set(mapped))
        return mapping_json
    except Exception as e:
        logger.error(f'Exception in create_mapping_json : {e}')


def reallocate_temps(mapping_json, temp_ids, logger):
    
    #Assign smallest temp_id for a cluster  
    def create_map(lists):
        curr_pool = [lis for lis in lists if lis not in pred_pool]
        if not curr_pool:
            return 1
        pred_pool.extend(curr_pool)
        for temps in curr_pool:
            create_map(mapping_json[temps] + [key for key in mapping_json if temps in mapping_json[key]])
    try:
        t_cluster = []
        for key in mapping_json:
            if key in sum(t_cluster, []):
                continue
            pred_pool = [key]
            create_map(mapping_json[key])
            if len(pred_pool):
                t_cluster.append(sorted(pred_pool))
        consol_temps = {}
        flag = 0
        for clust in t_cluster:
            if len(clust) > 1:
                if 20000 in clust:
                    flag = 1
                base = max(clust) if flag else clust[0]
                flag = 0
                consol_temps[base] = temp_ids[base]
                for img in clust:
                    if img == base:
                        continue
                    consol_temps[base] += temp_ids[img]
            else:
                consol_temps[clust[0]] = temp_ids[clust[0]]

        junk=20000
        if junk not in consol_temps and junk in temp_ids:
            consol_temps[junk] = temp_ids[junk]
        elif junk not in consol_temps and junk not in temp_ids:
            consol_temps[junk]=[]
        for key in temp_ids:
            if key not in sum(t_cluster,[]) and key != 20000:
                consol_temps[junk] += temp_ids[key]
        return consol_temps

    except Exception as e:
        logger.error(f'Exception in reallocate_temps : {e}')


def recreate_master_json(temp_ids, store_id, json_path, logger):
    try:
        master = {img: temp for temp in temp_ids.keys() for img in temp_ids[temp]}
        consolidated_master_json = dict(sorted(master.items(), key=lambda x: x[0]))
        re_path = f"{config['jsonPath']}/{date}/{store_id}/"
        os.makedirs(re_path, exist_ok=True)
        with open(json_path, 'r') as infile:
            temp_id_json = json.load(infile)
        consolidated_master_json['zipfile_name'] = temp_id_json['zipfile_name']
        consolidated_master_json['zipfile_count'] = temp_id_json['zipfile_count']
        # with open(f'{re_path}/consolidated_master.json', 'w') as fp:
        #     json.dump(consolidated_master_json, fp)
        logger.info(f'consolidated Master json created for {store_id}')
        return consolidated_master_json
    except Exception as e:
        print(e)
        logger.error(f'Exception in recreate_master_json : {e}')


def main_function(store_id):
    start = time.time()
    log_filename = f"{config['log_path_server1']}/consolidation_{current_process().name}.log"
    logging.getLogger().setLevel(logging.ERROR)
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(log_filename, when="midnight", backupCount=4)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    json_path = f"{config['jsonPath']}/{date}/{store_id}/master.json"
    mat_path = f"{config['matPath']}/{date}/{store_id}/emp_junk_person_features.mat"
    start = time.time()
    logger.info(f'generating input from mat files for {store_id}')
    for i in range(0, 5):
        try:
            with open(json_path, 'r') as infile:
                temp_id_json = json.load(infile)
            cust_ind = int([*temp_id_json.keys()][0])
            zipfile_names = temp_id_json['zipfile_name']
            del temp_id_json['zipfile_name']
            del temp_id_json['zipfile_count']
            cust_ind_end = int([*temp_id_json.keys()][-1])
            gallery_feature = scipy.io.loadmat(mat_path)
            gallery_feature = gallery_feature['emp_junk'][cust_ind - 1:cust_ind_end]
            break
        except Exception as error:
            time.sleep(1)
            logger.exception(f'Exception while accessing json/{error}')
            continue
    try:
        df=mat.main(gallery_feature,cust_ind,temp_id_json,store_id,logger)
        data = df.T.to_dict()
        temp_ids = {i: v.to_list() for i, v in df.groupby('temp_id').groups.items()}
    except Exception as e:
        logger.error(f'Exception in generating master_df {store_id} : {e}')
        for i in range(0, 3):
            try:
                df=mat.main(gallery_feature,cust_ind,temp_id_json,store_id,logger) 
                data = df.T.to_dict()
                temp_ids = {i: v.to_list() for i, v in df.groupby('temp_id').groups.items()}  
                break
            except Exception as e:
                logger.error(f'Retrying count {i+1}')
                continue        
    try:    
        before_count = 0
        for key in temp_ids:
            if key < 10000:
                before_count += 1
        logger.info(f'Before count for {store_id} : {before_count}')
        check_1=time.time()
        logger.info(f'time for inputs : {check_1-start}')
        final_temps = cluster_within_tempid(df, temp_ids, data, before_count,logger)
        check_2=time.time()
        logger.info(f'time for final_temps : {check_2-check_1}')
        mapping_json = create_mapping_json(temp_id_json,df, final_temps, data,gallery_feature,cust_ind,store_id, logger)
        check_3=time.time()
        logger.info(f'time for final_temps : {check_3-check_1}')
        combined_temp_ids = reallocate_temps(mapping_json, temp_ids, logger)
        combined_count = 0
        for key in combined_temp_ids:
            if key < 10000:
                combined_count += 1
        consol_master_json = recreate_master_json(combined_temp_ids, store_id, json_path, logger)
    except Exception as e:
        combined_count=0
        t_cluster=0
        logger.error(f'Exception in consolidation,Triggering reduction with master json {store_id}: {e}')
        with open(json_path, 'r') as infile:
            consol_master_json = json.load(infile)       
    logger.info(f'Consolidated count for {store_id} : {combined_count}')
    
    del df
    del data
    del temp_id_json
    gc.collect()
    
    audit_flag,consol_master_json=red_obj.main(date,store_id,records_df,gallery_feature,cust_ind,consol_master_json,logger)
    response=audit_prep.audit_df_gen(date, store_id,zipfile_names,cust_ind,consol_master_json,audit_flag,logger)
    response=1
    if response:
        logger.info(f'Audit triggered Succesfully for {date}/{store_id}')
    else:
        logger.error(f'Audit not triggered for {date}/{store_id}')
    logger.info(f'Time taken to complete for {store_id} : {time.time() - start} secs')

def main_test(store_id):
    return 1


if __name__ == '__main__':

    time_now=datetime.now()+timedelta(hours=5,minutes=30)
    date=datetime.strftime(time_now-timedelta(days=1),'%d-%m-%Y')
    # date='25-05-2022'
    today=datetime.strptime(date,'%d-%m-%Y')
    direc = f"{config['log_path_server1']}/"
    os.makedirs(direc, exist_ok=True)
    log_filename = f'{direc}/main_log.log'
    logging.getLogger().setLevel(logging.ERROR)
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(log_filename, when="midnight", backupCount=4)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    #fetching google sheet DataFrame
    for i in range(0, 5):
        try:
            logger.info('fetching google sheet..')
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(f'{scriptdir}/report-4da4f3d876e7.json', scope)
            client = gspread.authorize(creds)
            sheet = client.open('onboarding_stores')
            sheet_instance = sheet.get_worksheet(1)
            records_data = sheet_instance.get_all_records()
            records_df = pd.DataFrame.from_dict(records_data)
            # rs_obj = RedshiftDB(logger)
            # weekday_flag=today.weekday() < 5
            # if weekday_flag:
            #     dateslist = [datetime.strftime(today - timedelta(days = day),'%Y-%m-%d') for day in range(1,20) if (today - timedelta(days = day)).weekday() < 5]
            # else:
            #     dateslist = [datetime.strftime(today - timedelta(days = day),'%Y-%m-%d') for day in range(1,50) if (today - timedelta(days = day)).weekday() > 4]
            # footfall_df=rs_obj.execute_query(11,tuple(dateslist))
            logger.info(f'data collected')
            break
        except Exception as e:
            logger.error(f'Error in connecting to google sheet or redshift connection retrying..{e}')
            time.sleep(3)
            continue
    
    start = time.time()
    global_path = f"{config['jsonPath']}/{date}/"
    list_paths = [path for path in os.listdir(global_path) if path.startswith('11-')] 
    list_paths = ['11-187','11-763','11-788','11-860','11-867','11-949','11-961','11-969','11-440']
    batch_size = 3
    processName = [f'process_{i}' for i in range(1, batch_size + 1)]
    process = []
    for idx, proc in enumerate(list_paths):
        if idx < batch_size:
            logger.info(f'generating input from mat files for {proc}')
            p = multiprocessing.Process(name=processName[idx], target=main_function, args=(proc,))
            p.start()
            process.append(p)
        else:
            while (all([p.is_alive() for p in process])):
                time.sleep(1)
            for p in process:
                if not (p.is_alive()):
                    process.remove(p)
                    p = multiprocessing.Process(name=p.name, target=main_function, args=(proc,))
                    p.start()
                    process.append(p)
                    break
    for p in process:
        p.join()
    
    # data={"server_details": "consolidation_server1","status": "stop"}
    # try:
    #     response = sqs_resource.send_message(QueueUrl=f"https://sqs.us-east-1.amazonaws.com/871939287169/{config['status_message']}",
    #                                          MessageBody=json.dumps(data)
    #                                         )
    #     logger.info(f'Message Sent to SQS queue successfuly')
    # except Exception as e:
    #     logger.error(f'Problem sending completion message to SQS queue :{e}')    

    print(f'Time Taken to complete All Stores {(time.time() - start) / 60} Mins')
    logger.info(f'Time Taken to complete All Stores {(time.time() - start) / 60} Mins')