import numpy as np
import matplotlib.pyplot as plt
import core.misc as misc
import os
import core.LatticeAnalysis as LatticeAnalysis

class LatticeAnalysisUnit:
    def __init__(self) :
        #self.FolderPath = r"C:\Users\Junhyeok\Desktop\Project\FlourImage\GoodMott"
        self.PhaseLogPath = r"C:\Users\Junhyeok\Desktop\Project\FlourImage"


        self.PSF, self.lattice_a = misc.loadLatticeData(r"C:\Users\Junhyeok\Desktop\Project\LabAutomation\FlourAnalysis\FlourAnalysisMatlab\AnalysisData221210.mat")
        #newfile = misc.get_newest_file(FolderPath, ".dat")

        self.n = 2
        self.ang1 = -7.5 * np.pi / 180.0
        self.ang2 = 82.5 * np.pi / 180.0
        self.phaselog = {'x':[],'y':[]}
        self.LockPoint = {'x':0,'y':0}
        self.ROI_PSF = [150,980,1120]
        self.ROI_main = [200,250,450]


        self.params = misc.GenerateAnalysisParameters(self.ROI_main, self.lattice_a, self.PSF)
        self.params['LockPoint'] = self.LockPoint

        self.lattanalysis = LatticeAnalysis.LatticeAnalysis(self.params)

    def AnalysisOne(self, path) :     
        Latsum = 0
        phasexlog = []
        phaseylog = []
        Lats, photoncounts, xopts = self.lattanalysis.ProcessLattice(path)
        return Lats, photoncounts, xopts
    
    def AnalysisAll(self, Folderpath, fnames) : 
        for fname in fnames :
            Lats, photoncounts, xopts = self.lattanalysis.ProcessLattice(os.path.join(Folderpath, fname))
        return Lats, photoncounts, xopts



import influxdb_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import serial
import time
import datetime

     # timestamp = datetime.datetime.now(datetime.timezone.utc) 


def SendQueryMachineTemp(measure_point, probe_type, timestamp,temperature) :
    token = "IP7opg_k5MF_gSaFVaCudqQIjyjVj5fQorzxqzC-JAfxYnotppgHrF9a-DNM3YpOuBaNV_d7_S2PBR5CXjn0dw=="
    url = "http://192.168.0.58:8086"
    org = "JYLab3F"
    bucket="FlourAnalysis"
    point = (
        Point("LatticeAnalysis")
        .tag("Location",measure_point)
        .tag("Type",probe_type)
        .time(timestamp)
        .field("Celsius",temperature)        
    )
    #point = (
    #    Point("measurement1")
    #    .tag("tagname1", "tagvalue1")
    #    .field("field1", 0)
    #)
    write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    write_api = write_client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=bucket, org  = "JYLab3F", record=point)
    #query_result = client.query('SELECT * FROM "MachineTemp"."autogen"."temperature"')
    #print(query_result)
    query_result = ""
    write_client.close()

    return query_result