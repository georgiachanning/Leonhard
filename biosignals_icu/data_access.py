"""
Copyright (C) 2019 ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function

import sqlite3
from os.path import join
from program_args import Parameters


class DataAccess(object):
    DB_FILE_NAME = "mimic3.db"

    def __init__(self, data_dir):
        self.db = self.connect(data_dir)
        self.program_args = Parameters.parse_parameters()
        self.limit_parameter = int(self.program_args["num_patients_to_load"])
        self.offset_parameter = int(self.program_args["offset"])
        self.loaded_patients = self.get_patients()
        loaded_patients_interim = map(str, self.loaded_patients)
        self.loaded_patients_string = ", ".join(loaded_patients_interim)

    def connect(self, data_dir):
        db = sqlite3.connect(join(data_dir, DataAccess.DB_FILE_NAME),
                             check_same_thread=False,
                             detect_types=sqlite3.PARSE_DECLTYPES)
        return db

    @staticmethod
    def get_sampling_rate(device_and_signal_name):
        rates_dict = {  # In Hz
            "cns_ap-mean": 2.5,
            "cns_ap-syst": 2.5,
            "cns_ap-dias": 2.5,
            "cns_ap-na": 2.5,
            "cns_ci-na": 2.5,
            "cns_co-na": 2.5,
            "cns_cpi-na": 2.5,
            "cns_cpo-na": 2.5,
            "cns_dpmx-na": 2.5,
            "cns_elwi-na": 2.5,
            "cns_evlw-na": 2.5,
            "cns_gedi-na": 2.5,
            "cns_gedv-na": 2.5,
            "cns_gef-na": 2.5,
            "cns_itbi-na": 2.5,
            "cns_itbv-na": 2.5,
            "cns_pcco-na": 2.5,
            "cns_ppv-na": 2.5,
            "cns_pvpi-na": 2.5,
            "cns_sv-na": 2.5,
            "cns_svi-na": 2.5,
            "cns_svr-na": 2.5,
            "cns_svv-na": 2.5,
            "cns_art-mean": 1,
            "cns_art-syst": 1,
            "cns_art-dias": 1,
            "cns_pbto2-na": 0.5,
            "cns_ict-na": 0.5,
            "cns_cpp-na": 1,
            "cns_cpp2-na": 1,
            "cns_cvp-mean": 1,
            "cns_icp-mean": 1,
            "cns_icp2-mean": 1,
            "cns_hr-na": 1,
            "cns_lap-mean": 1,
            "cns_rr-na": 1,
            "cns_spo2-na": 1,
            "draeger_art": 100,
            "draeger_cvp": 100,
            "draeger_spo2": 100,
            "draeger_resp": 100,
            "draeger_icp": 100,
            "draeger_icp2": 100,
            "cns_cstat-na": 1,
            "cns_etco2-na": 1,
            "cns_expminvol-na": 1,
            "cns_fio2-na": 1,
            "cns_ftotal-na": 1,
            "cns_peep-na": 1,
            "cns_pinsp-na": 1,
            "cns_pmean-na": 1,
            "cns_pminimum-na": 1,
            "cns_ppeak-na": 1,
            "cns_pplateau-na": 1,
            "cns_rinsp-na": 1,
            "cns_rsb-na": 1,
            "cns_te-na": 1,
            "cns_ti-na": 1,
            "cns_fspontpct-na": 1,
            "cns_vte-na": 1,
            "cns_tinfinity-a": 1,
            "cns_tinfinity-b": 1,
            "cns_nbp-mean": 1,
        }
        if device_and_signal_name in rates_dict:
            return rates_dict[device_and_signal_name]
        else:
            return None

    def has_table(self, table_name):
        return self.db.execute("SELECT name "
                               "FROM sqlite_master "
                               "WHERE type='table' AND name='{table_name}';"
                               .format(table_name=table_name))\
                   .fetchone() is not None

    def get_num_rows(self, table_name):
        # This query assumes that there has not been any deletions in the time series table.
        return self.db.execute("SELECT MAX(_ROWID_) "
                               "FROM '{table_name}' "
                               "LIMIT 1;"
                               .format(table_name=table_name))\
                      .fetchone()[0]

    def get_all_table_names(self):
        return map(lambda x: x[0], self.db.execute("SELECT name FROM 'sqlite_master' WHERE type='table';").fetchall())

    def get_patients(self):

        if self.program_args["get_kids"]:
            if self.limit_parameter is None:
                patient_list_in_tuple = list(self.db.execute("SELECT DISTINCT subject_id FROM PATIENTS "
                                                             "ORDER BY subject_id;").fetchall())
            else:
                patient_list_in_tuple = list(self.db.execute("SELECT DISTINCT subject_id FROM PATIENTS ORDER BY subject_id "
                                                             "LIMIT {limit} OFFSET {offset};"
                                                             .format(limit=self.limit_parameter,
                                                                     offset=self.offset_parameter)))
        else:
            if self.limit_parameter is None:
                patient_list_in_tuple = list(self.db.execute("SELECT DISTINCT PATIENTS.SUBJECT_ID FROM PATIENTS "
                                                             "INNER JOIN ADMISSIONS ON "
                                                             "ADMISSIONS.SUBJECT_ID = PATIENTS.SUBJECT_ID "
                                                             "WHERE julianday(ADMISSIONS.admittime) - "
                                                             "julianday(PATIENTS.DOB) > 5840 "
                                                             "ORDER BY PATIENTS.subject_id ;"))
            else:
                patient_list_in_tuple = list(self.db.execute("SELECT DISTINCT PATIENTS.SUBJECT_ID FROM PATIENTS "
                                                             "INNER JOIN ADMISSIONS ON "
                                                             "ADMISSIONS.SUBJECT_ID = PATIENTS.SUBJECT_ID "
                                                             "WHERE julianday(ADMISSIONS.admittime) - "
                                                             "julianday(PATIENTS.DOB) > 5840 "
                                                             "ORDER BY PATIENTS.subject_id "
                                                             "LIMIT {limit} OFFSET {offset};"
                                                             .format(limit=self.limit_parameter,
                                                                     offset=self.offset_parameter)))

        patient_array_list = list(sum(patient_list_in_tuple, ()))
        return patient_array_list

    def get_last_timestamp(self, table_name):
        result = self.db.execute("SELECT timestamp "
                                 "FROM '{table_name}' "
                                 "WHERE _ROWID_ = (SELECT MAX(_ROWID_) FROM '{table_name}');"
                                 .format(table_name=table_name)).fetchone()

        return result if result is None else result[0]

    def get_items_by_id_set(self, id_set=set(), table_name="CHARTEVENTS",
                            get_subjects=False, value_case=None):

        if get_subjects:
            columns = "DISTINCT subject_id, charttime, valuenum"
        else:
            columns = "charttime, valuenum" \

        id_set = map(str, id_set)
        id_set_string = ", ".join(id_set)
        data = []

        for patient in self.loaded_patients:
            query_per_patient = ("SELECT {COLUMNS} "
                                 "FROM {TABLE_NAME} "
                                 "WHERE subject_id = {patient} AND ITEMID IN ({ID_SET}); ")\
                .format(COLUMNS=columns, TABLE_NAME=table_name,
                        ID_SET=id_set_string,
                        offset=self.offset_parameter,
                        limit=self.limit_parameter,
                        patient=patient)
            data.append(self.db.execute(query_per_patient).fetchall())

        flat_data = [item for sublist in data for item in sublist]

        return flat_data

    def get_items_by_drug(self, id_set=set(), table_name="PRESCRIPTIONS",
                         get_subjects=True, value_case=None):
        if get_subjects:
            columns = "DISTINCT subject_id, startdate, enddate"
        else:
            columns = "subject_id, drug" \
                if value_case is None else \
                "subject_id, drug".format(VALUE_CASE=value_case)

        id_set = map(str, id_set)
        id_set_string = ", ".join(id_set)
        data = []

        for patient in self.loaded_patients:
            query_per_patient = ("SELECT {COLUMNS} "
                                 "FROM {TABLE_NAME} "
                                 "WHERE subject_id = {patient} AND DRUG IN ({ID_SET}); ")\
                .format(COLUMNS=columns, TABLE_NAME=table_name,
                        ID_SET=id_set_string,
                        offset=self.offset_parameter,
                        limit=self.limit_parameter,
                        patient=patient)
            data.append(self.db.execute(query_per_patient).fetchall())

        flat_data = [item for sublist in data for item in sublist]

        return flat_data

    def get_items_by_icd(self, id_set=set(), table_name="DIAGNOSES_ICD",
                         get_subjects=False, value_case=None):
        if get_subjects:
            columns = "DISTINCT subject_id"
        else:
            columns = "subject_id, icd9_code" \
                if value_case is None else \
                "subject_id, icd9_code".format(VALUE_CASE=value_case)

        id_set = map(lambda x: '"' + str(x) + '"', id_set)
        id_set_string = ", ".join(id_set)
        data = []

        for patient in self.loaded_patients:
            query_per_patient = ("SELECT {COLUMNS} "
                                 "FROM {TABLE_NAME} "
                                 "WHERE subject_id = {patient} AND ICD9_CODE IN ({ID_SET}); ")\
                .format(COLUMNS=columns, TABLE_NAME=table_name,
                        ID_SET=id_set_string,
                        offset=self.offset_parameter,
                        limit=self.limit_parameter,
                        patient=patient)
            data.append(self.db.execute(query_per_patient).fetchall())

        flat_data = [item for sublist in data for item in sublist]

        return list(sum(flat_data, ()))

    def get_spo2_values(self, patient_id):
        # From: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day-arterial.sql
        spo2_item_ids = {
            "646",
            "220277"
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=spo2_item_ids)

    def get_etco2_values(self, patient_id):
        etco2_item_ids = {
            "1817",
            "228640",
            "228641"
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=etco2_item_ids)

    def get_fio2_values(self, patient_id):
        # From: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day-arterial.sql
        fio2_item_ids = {
            "3420",    # FiO2
            "190",     # FiO2 set
            "223835",  # Inspired O2 Fraction (FiO2)
            "3422"     # FiO2 [measured]
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=fio2_item_ids,
                                        value_case="case"
                                                   "    when itemid = 223835 "
                                                   "      then case "
                                                   "        when valuenum > 0 and valuenum <= 1 "
                                                   "          then valuenum * 100 "
                                                   "        when valuenum > 1 and valuenum < 21 "
                                                   "          then null "
                                                   "        when valuenum >= 21 and valuenum <= 100 "
                                                   "          then valuenum "
                                                   "        else null end "
                                                   "  when itemid in (3420, 3422) "
                                                   "      then valuenum "
                                                   "  when itemid = 190 and valuenum > 0.20 and valuenum < 1 "
                                                   "      then valuenum * 100 "
                                                   "else null end as valuenum ")

    def get_peep_values(self, patient_id):
        # From: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/durations/ventilation-durations.sql
        peep_item_ids = {"60", "437", "505", "506", "686", "220339", "224700"}
        return self.get_items_by_id_set(patient_id=patient_id, id_set=peep_item_ids)

    def get_hr_values(self, patient_id):
        item_ids = {
            211, 220045
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_sysbp_values(self, patient_id):
        item_ids = {
            51, 442, 455, 6701, 220179, 220050
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_diasbp_values(self, patient_id):
        item_ids = {
            8368, 8440, 8441, 8555, 220180, 220051
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_meanbp_values(self, patient_id):
        item_ids = {
            456, 52, 6702, 443, 220052, 220181, 225312
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_rr_values(self, patient_id):
        item_ids = {
            615, 618, 220210, 224690
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_tempc_values(self, patient_id):
        item_ids = {
            223762, 676,
            223761, 678
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids,
                                        value_case="case when itemid in (223761, 678) "
                                                   "then(valuenum - 32) / 1.8 else valuenum end as valuenum")

    def get_icp_values(self, patient_id):
        item_ids = {
            226, 220765
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_cpp_values(self, patient_id):
        item_ids = {
            227066,
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_tidal_volume_values(self, patient_id):
        item_ids = {
            639, 654, 681, 682, 683, 684, 224685, 224684, 224686
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_inspiratory_time_values(self, patient_id):
        item_ids = {
            224738, 2000
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_resistance_values(self, patient_id):
        item_ids = {
            220283
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_plateau_pressure_values(self, patient_id):
        item_ids = {
            543, 224696
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_min_pressure_values(self, patient_id):
        item_ids = {
            436, 3143, 6864, 6552, 7176
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_mean_pressure_values(self, patient_id):
        item_ids = {
            444, 224697
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_peak_pressure_values(self, patient_id):
        item_ids = {
            218, 535, 224695, 1686, 6047, 507
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_pinsp_values(self, patient_id):
        item_ids = {
            227187
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_ftotal_values(self, patient_id):
        item_ids = {
            619, 224688
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=item_ids)

    def get_pao2_values(self, patient_id):
        # From: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day.sql
        bloodgas_item_ids = {
            "50821",  # PaO2
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=bloodgas_item_ids, table_name="labevents")

    def get_fio2_lab_values(self, patient_id):
        # From: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day.sql
        bloodgas_item_ids = {
            "50816",  # FiO2
        }
        return self.get_items_by_id_set(patient_id=patient_id, id_set=bloodgas_item_ids, table_name="labevents",
                                        value_case="case when valuenum < 20 then null "
                                                   " when valuenum > 100 then null "
                                                   " else valuenum end as valuenum")

    def get_rrates(self):
        item_ids = {
            3024171, 44818701, 8541, 615, 1635, 1151, 2117, 3603, 3337, 1884, 618, 220210, 224690
        }
        return self.get_items_by_id_set(get_subjects=True, id_set=item_ids)

    def get_blood_sugar(self):  # TODO: WRONG IDS
        item_ids = {
            3024171, 44818701, 8541, 615, 1635, 1151, 2117, 3603, 3337, 1884, 456, 52, 6702, 443, 220052, 220181, 225312
        }
        return self.get_items_by_id_set(get_subjects=True, id_set=item_ids)

    def get_sodium(self):
        item_ids = {
            50824, 50983,
        }
        return self.get_items_by_id_set(get_subjects=True, id_set=item_ids, table_name="LABEVENTS")

    def get_heart_rate(self):
        item_ids = {
            51, 442, 455, 6701, 220179, 220050, 211, 220045
        }
        return self.get_items_by_id_set(get_subjects=True, id_set=item_ids)

    def get_glucose(self):
        item_ids = {
            807, 811, 1529, 3745, 3744, 225664, 220621, 226537,
        }
        return self.get_items_by_id_set(get_subjects=True, id_set=item_ids)

    def get_mean_blood_pressure(self):
        item_ids = {
            456, 52, 6702, 443, 220052, 220181, 225312,
        }
        return self.get_items_by_id_set(get_subjects=True, id_set=item_ids)

    def get_admit_time(self):
        admit_times = {}
        patients_with_arrhythmias = self.get_patients_with_arrhythmias()
        query = ( "SELECT SUBJECT_ID, ADMITTIME, HADM_ID "
                  "FROM ADMISSIONS "
                  "WHERE ADMISSIONS.subject_id IN ({loaded_patients}) "
                  "ORDER BY ADMITTIME DESC ;").format(loaded_patients=self.loaded_patients_string)
        # arranged in descending order because when put into dictionary order will be reversed

        all_admit_times = self.db.execute(query).fetchall()

        for patient in self.loaded_patients:
            for admittime in range(len(all_admit_times)):
                if patient == all_admit_times[admittime][0]:
                    admit_times[patient] = all_admit_times[admittime][1]

            if patient in patients_with_arrhythmias:
                query = ("SELECT ADMISSIONS.ADMITTIME FROM ADMISSIONS "
                         "INNER JOIN DIAGNOSES_ICD ON "
                         "DIAGNOSES_ICD.HADM_ID = ADMISSIONS.HADM_ID "
                         "WHERE ADMISSIONS.SUBJECT_ID = '{patient_id}' AND "
                         "DIAGNOSES_ICD.icd9_code IN (42610 , 42611, 42613, 4262, "
                         "42653, 4266, 42689, 4270, 4272, 42731, 42760, 4279, 7850);").format(patient_id=patient)
                admit_times[patient] = self.db.execute(query).fetchone()[0]

        assert len(self.loaded_patients) == len(admit_times)

        return admit_times

    def get_potassium(self):
        item_ids = {
            3725, 1535, 829, 50883, 50971, 44711, 50822

        }
        return self.get_items_by_id_set(get_subjects=True, id_set=item_ids, table_name="LABEVENTS")

    # when icd9_code = '42610' then 1
    #     when icd9_code = '42611' then 1
    #     when icd9_code = '42613' then 1
    #     when icd9_code between '4262 ' and '42653' then 1
    #     when icd9_code between '4266 ' and '42689' then 1
    #     when icd9_code = '4270 ' then 1
    #     when icd9_code = '4272 ' then 1
    #     when icd9_code = '42731' then 1
    #     when icd9_code = '42760' then 1
    #     when icd9_code = '4279 ' then 1
    #     when icd9_code = '7850 ' then 1
    #     when icd9_code between 'V450 ' and 'V4509' then 1
    #     when icd9_code between 'V533 ' and 'V5339' then 1
    #current get by icd function wont let me use the ones with letters

    def get_patients_with_arrhythmias(self):
        item_ids = {
            "42610", "42611", "42613", "4262", "42653", "4266", "42689", "4270", "4272", "42731", "42760", "4279",
            "7850"
        }
        patients_with_arrhythmias = self.get_items_by_icd(get_subjects=True, id_set=item_ids)
        return patients_with_arrhythmias

    def get_patients_with_dyspnea(self):
        item_ids = {
            "R06.0", "R06.00", "R06.06",
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_cocaine(self):
        item_ids = {
            "F14.0", "F14.1", "F14.2", "F14.20", "F14.21", "F14.22", "F14.23", "F14.24", "F14.28", "F14.29",
            "F14.25", "F14.9", "R78.2",
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_cardiac_arrest(self):
        item_ids = {
            "I46.2", "I46.8", "I46.9",
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_chest_pain(self):
        item_ids = {
            "R07.1", "R07.8", "R07.89", "R07.9",
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_orthopnea(self):
        item_ids = {
            "R06.01",
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_epilepsy_history(self):
        item_ids = {
            "G40.0", "G40.1", "G40.2", "G40.3", "G40.4", "G40.8", "G40.8", "G40.B", "Z82.0",
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_alcohol_abuse(self):
        item_ids = {
            "F10.1", '980', '2652','2911','2912','2913',
            '2915','2918','2919','3030','3039', '3050', '3575', '4255', '5353', '5710', '5711', '5712',' 5713', 'V113',
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_muscular_dystrophy(self):
        item_ids = {
            "G71.0", "G71.11",
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_heart_failure(self):
        item_ids = {
            "I50", '39891','40201','40211','40291','40401','40403',
            '40411','40413','40491','40493', '4254','4255','4257','4258','4259', '428'
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_renal_failure(self):
        item_ids = {
            '40301','40311','40391','40402','40403','40412','40413','40492','40493',
            '5880','V420','V451', '585','586','V56',
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_lung_disease(self):
        item_ids = {
            '4168', '4169', '5064', '5081', '5088',
            '490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505',
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_patients_with_pulmonary_circulation_disorder(self):
        item_ids = {
            '4150', '4151', '4170', '4178', '4179', '416'
        }
        return self.get_items_by_icd(get_subjects=True, id_set=item_ids)

    def get_dob(self):
        patient_and_dob = self.db.execute("SELECT subject_id, dob "
                                          "FROM PATIENTS "
                                          "ORDER BY subject_id ;"
                                             ).fetchall()
        return map(lambda x: x[0], patient_and_dob)

    def get_gender(self):
        patient_and_gender = self.db.execute("SELECT subject_id, gender "
                                             "FROM PATIENTS "
                                             "ORDER BY subject_id ;"
                                                ).fetchall()
        return map(lambda x: x[0], patient_and_gender)

    def get_patients_with_terfenadine(self):
        terfenadine = self.db.execute("SELECT subject_id, startdate, enddate "
                                      "FROM PRESCRIPTIONS "
                                      "WHERE DRUG LIKE 'Terfenadine' "
                                      "ORDER BY subject_id ;").fetchall()
        return map(lambda x: x[0], terfenadine)

    def get_patients_with_astemizole(self):
        astemizole = self.db.execute("SELECT subject_id, startdate, enddate "
                                     "FROM PRESCRIPTIONS "
                                     "WHERE DRUG LIKE 'Astemizole' "
                                     "ORDER BY subject_id ;").fetchall()
        return map(lambda x: x[0], astemizole)

    def get_patients_with_quinine(self):
        quinine = self.db.execute("SELECT subject_id, startdate, enddate "
                                  "FROM PRESCRIPTIONS "
                                  "WHERE DRUG LIKE 'Quinine' OR 'Quinoline'"
                                  "ORDER BY subject_id ;").fetchall()
        return map(lambda x: x[0], quinine)



