
import pickle
import json
import azureml.train.automl
import pandas as pd
from sklearn.externals import joblib
from azureml.core.model import Model


def init():
    global model
    model_path = Model.get_model_path(model_name='Capstone_Project')
    model = joblib.load(model_path)

def run(rawdata):
    try:
        data = json.loads(rawdata)['data']
        df = pd.DataFrame(data,columns=["row_id","country","country_ID","is_urban","age","age_grp","female","married","religion","religion_Id","relationship_to_hh_head","relationship_to_hh_head_ID","education_level","literacy","can_add","can_divide","can_calc_percents","can_calc_compounding","employed_last_year","employment_category_last_year","employment_category_last_year_Id","employment_type_last_year","employment_type_last_year_Id","share_hh_income_provided","income_ag_livestock_last_year","income_friends_family_last_year","income_government_last_year","income_own_business_last_year","income_private_sector_last_year","income_public_sector_last_year","num_times_borrowed_last_year","borrowing_recency","formal_savings","informal_savings","cash_property_savings","has_insurance","has_investment","bank_interest_rate","mm_interest_rate","mfi_interest_rate","other_fsp_interest_rate","num_shocks_last_year","avg_shock_strength_last_year","borrowed_for_emergency_last_year","borrowed_for_daily_expenses_last_year","borrowed_for_home_or_biz_last_year","phone_technology","can_call","can_text","can_use_internet","can_make_transaction","phone_ownership","advanced_phone_use","reg_bank_acct","reg_mm_acct","reg_formal_nbfi_account","financially_included","active_bank_user","active_mm_user","active_formal_nbfi_user","active_informal_nbfi_user","nonreg_active_mm_user","num_formal_institutions_last_year","num_informal_institutions_last_year","num_financial_activities_last_year","type"])
        result = model.predict(df)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result": result.tolist()})
