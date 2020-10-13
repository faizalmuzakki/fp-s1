import mysql.connector
from random import uniform
import pandas as pd
from math import sin, cos, sqrt, atan2, radians

ppdb = mysql.connector.connect(
    host = "103.83.4.58",
    user = "input2018",
    passwd = "n4s1kuc1ngen4ksek4l1",
    database = "ppdb_sby_2019"
)

ppdb_cursor = ppdb.cursor(buffered=True)

ppdb2020 = mysql.connector.connect(
    host = "103.83.4.58",
    user = "input2018",
    passwd = "n4s1kuc1ngen4ksek4l1",
    database = "ppdb_sby_2020"
)

ppdb2020_cursor = ppdb2020.cursor(buffered=True)

ppdb2020out = mysql.connector.connect(
    host = "103.83.4.62",
    user = "input2018",
    passwd = "n4s1kuc1ngen4ksek4l1",
    database = "ppdb_sby_rank_2020"
)

ppdb2020out_cursor = ppdb2020out.cursor(buffered=True)

def getData():
    csv = open('data/siswa.csv', 'w')
    print('nik,lat,lng,lokasi,latsmp,lngsmp,jalur,jarak,diterima', file=csv)

    jalurs = ['zonasi']

    for jalur in jalurs:
        sql = "select mvalue from configs where parameter = %s limit 1"
        val = ('smp_' + jalur, )
        ppdb_cursor.execute(sql, val)
        mvalue = ppdb_cursor.fetchone()[0]

        if(jalur == 'pp'):
            tabel_input = 'pemenuhan_pagu'
        else:
            tabel_input = 'smp_'+jalur
        
        sql = """select i.input_nik,p.latitude,p.longitude,i.input_id_lokasi,o.output_pilihan1,o.output_pilihan2,o.output_jarak1,o.output_jarak2,o.output_diterima,output_pilihan
                from input_{} i
                join output_smp_{} o on i.input_nik = o.output_nik
                join ppdb_batasrt p on p.id_batasrt = i.input_id_lokasi""".format(tabel_input, jalur+mvalue)
        ppdb_cursor.execute(sql)
        rowcount = ppdb_cursor.rowcount
        res = ppdb_cursor.fetchall()

        mark = 1
        for row in res:
            print(str(mark) + '/' + str(rowcount))

            output_pilihan1 = row[4]
            output_pilihan2 = row[5]
            output_jarak1 = row[6]
            output_jarak2 = row[7]
            
            if(int(row[9]) == 2):
                jarak = output_jarak2
                smp = int(output_pilihan2) - 100
            else:
                jarak = output_jarak1
                smp = int(output_pilihan1) - 100

            sql = "select latitude,longitude from ppdb_smpn where id_smpn = %s limit 1"
            val = (str(smp), )
            ppdb_cursor.execute(sql, val)
            smp_detail = ppdb_cursor.fetchone()

            latsmp = str(smp_detail[0])
            lngsmp = str(smp_detail[1])

            data = "{},{},{},{},{},{},{},{},{}".format(row[0],row[1],row[2],row[3],latsmp,lngsmp,jalur,jarak,smp)

            print(data, file=csv)
            mark += 1

def get_allData():
    csv = open('data/siswaall.csv', 'w')
    print('nik,lat,lng,lokasi', file=csv)

    jalurs = ['zonasi']

    for jalur in jalurs:
        sql = "select mvalue from configs where parameter = %s limit 1"
        val = ('smp_' + jalur, )
        ppdb_cursor.execute(sql, val)
        mvalue = ppdb_cursor.fetchone()[0]

        if(jalur == 'pp'):
            tabel_input = 'pemenuhan_pagu'
        else:
            tabel_input = 'smp_'+jalur
        
        sql = """select i.input_nik,p.latitude,p.longitude,i.input_id_lokasi
                from input_{} i
                join ppdb_batasrt p on p.id_batasrt = i.input_id_lokasi""".format(tabel_input, jalur+mvalue)
        ppdb_cursor.execute(sql)
        rowcount = ppdb_cursor.rowcount
        res = ppdb_cursor.fetchall()

        mark = 1
        for row in res:
            print(mark, rowcount)
            data = "{},{},{},{}".format(row[0],row[1],row[2],row[3])

            print(data, file=csv)
            mark += 1

def newpoint():
   return uniform(-7.165,-7.36), uniform(112.58, 112.82)

def calculateDistance(lat, newlat, lng, newlng):
    R = 6373.0

    lat1 = radians(lat)
    lon1 = radians(lng)
    lat2 = radians(newlat)
    lon2 = radians(newlng)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c * 1000

    return distance

def getDataRandom500m():
    filepath = "/mnt/e/Projects/tugas-akhir/fp/data/siswa.csv"
    
    csv = open('data/siswa2.csv', 'w')
    print('nik,lat,lng,lokasi,jarak,diterima,latsmp,lngsmp,jalur', file=csv)

    i = 1
    df = pd.read_csv(filepath)
    for row in range(df.shape[0]):
        print(str(i) + "/" + str(df.shape[0]))
        i-=-1

        nik = df.iat[row, 0]
        lat = df.iat[row, 1]
        lng = df.iat[row, 2]
        lokasi = df.iat[row, 3]
        latsmp = df.iat[row, 4]
        lngsmp = df.iat[row, 5]
        jalur = df.iat[row, 6]
        jarak = df.iat[row, 7]
        diterima = df.iat[row, 8]

        newlat, newlng = newpoint()
        distance = calculateDistance(lat, newlat, lng, newlng)
        while(distance > 500):
            newlat, newlng = newpoint()
            distance = calculateDistance(lat, newlat, lng, newlng)

        data = "{},{},{},{},{},{},{},{},{}".format(nik,newlat,newlng,lokasi,distance,diterima,latsmp,lngsmp,jalur)
        print(data, file=csv)

def getSekolah():
    csv = open('data/sekolah.csv', 'w')
    print('kode,jml,peminat,lat,lng,radius', file=csv)

    sql = "select id_smpn, latitude, longitude from ppdb_smpn"
    ppdb_cursor.execute(sql)
    sekolahs = ppdb_cursor.fetchall()

    for sekolah in sekolahs:
        sql = "select kode_sekolah from sekolah where id_sekolah = '{}'".format(int(sekolah[0]) + 100)
        ppdb_cursor.execute(sql)
        kode_sekolah = ppdb_cursor.fetchone()[0]

        sql = "select count(*) from input_smp_zonasi where input_pilihan1 = %s or input_pilihan2 = %s"
        val = (sekolah[0] + 100, sekolah[0] + 100)
        ppdb_cursor.execute(sql, val)
        peminat = ppdb_cursor.fetchone()[0]

        sql = "select mvalue from configs where parameter = 'smp_zonasi' limit 1"
        ppdb_cursor.execute(sql)
        mvalue = ppdb_cursor.fetchone()[0]

        sql = """select output_pilihan1, output_pilihan2, output_jarak1, output_jarak2
                from output_smp_zonasi{}
                where output_diterima = {}""".format(mvalue, int(sekolah[0]) + 100)
        ppdb_cursor.execute(sql)
        rowcount = ppdb_cursor.rowcount
        res = ppdb_cursor.fetchall()
        
        count = len(res)

        radius = 0.0
        for row in res:
            output_pilihan1 = int(row[0])
            output_pilihan2 = int(row[1])
            output_jarak1 = float(row[2])
            output_jarak2 = float(row[3])

            if(output_pilihan1 == (int(sekolah[0]) + 100)):
                if(output_jarak1 > radius):
                    radius = output_jarak1
            elif(output_pilihan2 == (int(sekolah[0]) + 100)):
                if(output_jarak2 > radius):
                    radius = output_jarak2
        print("{},{},{},{},{}".format(kode_sekolah, count, peminat, sekolah[1], sekolah[2], radius), file=csv)

def getData2020():
    csv = open('data/2020/siswa.csv', 'w')
    print('nik,lat,lng,latsmp,lngsmp,jalur,jarak,diterima,lokasi', file=csv)

    csv2 = open('data/2020/siswaall.csv', 'w')
    print('nik,lat,lng,lokasi', file=csv2)

    jalurs = ['zonasi']

    for jalur in jalurs:
        sql = "select mvalue from configs where parameter = %s limit 1"
        val = ('smp_' + jalur, )
        ppdb2020out_cursor.execute(sql, val)
        mvalue = ppdb2020out_cursor.fetchone()[0]

        if(jalur == 'pp'):
            tabel_input = 'pemenuhan_pagu'
        else:
            tabel_input = 'smp_'+jalur

        sql = "select id_smpn, latitude, longitude from ppdb_smpn"
        ppdb2020_cursor.execute(sql)
        res = ppdb2020_cursor.fetchall()

        sekolahs = {}
        for sekolah in res:
            sekolahs[sekolah[0] + 100] = [sekolah[1], sekolah[2]]

        sql = """select i.input_nik, sd.sd_lat, sd.sd_lng
                from input_{} i
                join data_siswa_sd sd on sd.sd_nik = i.input_nik""".format(tabel_input)
        ppdb2020_cursor.execute(sql)
        rowcount = ppdb2020_cursor.rowcount
        res = ppdb2020_cursor.fetchall()

        mark = 1
        for row in res:
            print(str(mark) + '/' + str(rowcount))
            mark += 1

            nik = row[0]
            lat = row[1]
            lng = row[2]

            sql = """select output_pilihan1, output_pilihan2, output_jarak1, 
                output_jarak2, output_diterima, output_pilihan
                from output_smp_{}{}
                where output_nik = %s""".format(jalur, mvalue)
            val = (nik, )
            ppdb2020out_cursor.execute(sql, val)
            output = ppdb2020out_cursor.fetchone()

            output_pilihan1 = int(output[0])
            output_pilihan2 = int(output[1])
            output_jarak1 = output[2]
            output_jarak2 = output[3]
            output_diterima = int(output[4])

            print("{},{},{}".format(nik,lat,lng), file=csv2)
            
            if(output_diterima > 100):
                if(output_diterima == output_pilihan2):
                    jarak = output_jarak2
                else:
                    jarak = output_jarak1
            else:
                continue

            latsmp = str(sekolahs[output_diterima][0])
            lngsmp = str(sekolahs[output_diterima][1])

            data = "{},{},{},{},{},{},{},{}".format(nik,lat,lng,latsmp,lngsmp,jalur,jarak,output_diterima)

            print(data, file=csv)

def getSekolah2020():
    csv = open('data/2020/sekolah.csv', 'w')
    print('kode,jml,peminat,lat,lng,radius', file=csv)

    sql = "select id_smpn, latitude, longitude from ppdb_smpn"
    ppdb2020_cursor.execute(sql)
    sekolahs = ppdb2020_cursor.fetchall()

    for sekolah in sekolahs:
        sql = "select kode_sekolah from sekolah where id_sekolah = '{}'".format(int(sekolah[0]) + 100)
        ppdb2020_cursor.execute(sql)
        kode_sekolah = ppdb2020_cursor.fetchone()[0]

        sql = "select count(*) from input_smp_zonasi where input_pilihan1 = %s or input_pilihan2 = %s"
        val = (sekolah[0] + 100, sekolah[0] + 100)
        ppdb2020_cursor.execute(sql, val)
        peminat = ppdb2020_cursor.fetchone()[0]

        sql = "select mvalue from configs where parameter = 'smp_zonasi' limit 1"
        ppdb2020out_cursor.execute(sql)
        mvalue = ppdb2020out_cursor.fetchone()[0]

        sql = """select output_pilihan1, output_pilihan2, output_jarak1, output_jarak2
                from output_smp_zonasi{}
                where output_diterima = {}""".format(mvalue, int(sekolah[0]) + 100)
        ppdb2020out_cursor.execute(sql)
        rowcount = ppdb2020out_cursor.rowcount
        res = ppdb2020out_cursor.fetchall()
        
        count = len(res)

        radius = 0.0
        for row in res:
            output_pilihan1 = int(row[0])
            output_pilihan2 = int(row[1])
            output_jarak1 = float(row[2])
            output_jarak2 = float(row[3])

            if(output_pilihan1 == (int(sekolah[0]) + 100)):
                if(output_jarak1 > radius):
                    radius = output_jarak1
            elif(output_pilihan2 == (int(sekolah[0]) + 100)):
                if(output_jarak2 > radius):
                    radius = output_jarak2
        print("{},{},{},{},{}".format(kode_sekolah, count, peminat, sekolah[1], sekolah[2], radius), file=csv)

if __name__ == '__main__':
    getData2020()