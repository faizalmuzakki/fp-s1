from scipy.spatial import Voronoi
import numpy as np 
import random
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from csv import writer
from csv import reader
from sklearn.cluster import KMeans
import os
from math import sin, cos, sqrt, atan2, radians

import folium 
from folium import vector_layers 
import io
from folium.features import DivIcon
from ast import literal_eval
from PIL import Image, ImageChops
# from folium.plugins import HeatMap

def trim(im):
    """src: https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil """
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    
def show_map(m, filename = "voronoi.png"):
    """Must show folium maps as inline png to show on GitHub"""
    stream = io.BytesIO(m._to_png())
    img = Image.open(stream)
    
    # trim bc sometimes the folium.to_png() can generate whitespace 
    img = trim(img)
    
    # save instead of matplotlib.pyplot.imshow(np.img)) bc better quality
    img.save("result/" + filename)
    # png = Image_Display(filename='map.png')
    
    # clean up
    # Remove("m.png")
    # return png
    return

def plot_map(df, center=(-7.2732,112.7208), show_nums=True, show_seeds=True):
    m = folium.Map(location=[*center],
                   width=686, height=686, 
                   zoom_start=12,
                   api_key='6NbtVc32EkZBkf8eXLAE')
    
    for lat, lon, color, poly, jml, kode in df[["lat","lng","colors","polygons","jml","kode"]].values:
        points = to_convex(np.flip(poly).tolist())
        vlayer = vector_layers.Polygon(points, 
                                       fill=True, 
                                       color="black",
                                       fill_color="rgba({}, {}, {}, {})".format(*color),
                                       weight=1)
        m.add_child(vlayer)

        if show_seeds:
            clayer = vector_layers.Circle([lat,lon], 2, color="black")
            m.add_child(clayer)
        
        if show_nums:
            folium.Marker((lat, lon), icon=DivIcon(
            icon_size=(.1,.1),
            icon_anchor=(6,0),
            html='<div style="font-size: 8pt; color : black">%s</div>'%str(kode[3:]),
            )).add_to(m)
        
    return m

def plot_map2(center = (-7.2732,112.7208)):
    sekolahs = pd.read_csv("data/sekolah.csv")
    siswas = pd.read_csv("data/siswa_clustered.csv")
    clusters = pd.read_csv("data/siswa_centroids.csv")

    # "OpenStreetMap"
    # "Mapbox Bright" (Limited levels of zoom for free tiles)
    # "Mapbox Control Room" (Limited levels of zoom for free tiles)
    # "Stamen" (Terrain, Toner, and Watercolor)
    # "Cloudmade" (Must pass API key)
    # "Mapbox" (Must pass API key)
    # "CartoDB" (positron and dark_matter)

    m = folium.Map(location=[*center],
                   width=686, height=686, 
                   zoom_start=12,
                   tiles="OpenStreetMap")
                #    api_key='6NbtVc32EkZBkf8eXLAE'
    
    for lat, lon, color in siswas[["lat","lng","colors"]].values:
        color = literal_eval(color)
        clayer = vector_layers.Circle([lat,lon], 1, color="rgba({}, {}, {}, {})".format(*color))
        m.add_child(clayer)
    
    for lat, lon in clusters[["lat","lng"]].values:
        clayer = vector_layers.Circle([lat,lon], 3, color="red")
        m.add_child(clayer)

    for kode, lat, lon, radius in sekolahs[["kode","lat","lng","radius"]].values:
        # if(int(kode[3:]) == int(smp)):
        clayer = vector_layers.Circle([lat,lon], 3, color="black")
        m.add_child(clayer)
        # clayer = vector_layers.Circle([lat,lon], radius, color="black")
        # m.add_child(clayer)

        folium.Marker((lat, lon), icon=DivIcon(
        icon_size=(.1,.1),
        icon_anchor=(6,0),
        html='<div style="font-size: 8pt; color : black">%s</div>'%str(kode[3:]),
        )).add_to(m)
        
    return m

def to_convex(points):
    # compute centroid
    cent = (sum([p[0] for p in points])/len(points),
            sum([p[1] for p in points])/len(points))
    # sort by polar angle
    points.sort(key=lambda p: atan2(p[1] - cent[1],
                                    p[0] - cent[0]))
    return points

def generatePNG(filename = "siswa"):
    sekolahs = pd.read_csv("data/" + filename + "_centroids_voronoi.csv")
    sekolahs[["lat","lng"]] = sekolahs[["lat","lng"]].astype(float)

    # siswas = pd.read_csv("clustered.csv")
    # siswas = pd.read_csv("clustered2.csv")
    # siswas[["lat","lng"]] = siswas[["lat","lng"]].astype(float)

    # i = 0
    # for color in siswas["color"]:
    #     siswas["color"][i] = literal_eval(siswas["color"][i])
    #     i+=1

    # clusters = pd.read_csv("clusters.csv")
    # clusters[["lat","lng"]] = clusters[["lat","lng"]].astype(float)

    i = 0
    for colors in sekolahs["colors"]:
        sekolahs["colors"][i] = literal_eval(sekolahs["colors"][i])
        i+=1

    i = 0
    for polygons in sekolahs["polygons"]:
        sekolahs["polygons"][i] = literal_eval(sekolahs["polygons"][i])
        i+=1

    # # voronoi
    show_map(plot_map(sekolahs), filename + "_centroids_voronoi.png")

def random_color(as_str=True, alpha=0.5):
    rgb = [random.randint(0,255),
           random.randint(0,255),
           random.randint(0,255)]
    if as_str:
        return "rgba"+str(tuple(rgb+[alpha]))
    else:
        return list(np.array(rgb)/255) + [alpha]

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    Source
    -------
    Copied from https://gist.github.com/pv/8036995
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def voronoi_polygons(n=50):
    random_seeds = np.random.rand(n, 2)
    vor = Voronoi(random_seeds)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    polygons = []
    for reg in regions:
        polygon = vertices[reg]
        polygons.append(polygon)
    return polygons

# Add Voronoi cell polygons 
def calc_polygons(df):
    vor = Voronoi(df[["lng","lat"]].values)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    polygons = []
    for reg in regions:
        polygon = vertices[reg]
        polygons.append(polygon)
    return polygons

def nums_to_color(series, dots=False, cmap=cm.coolwarm_r, alpha=0.5):
    """
    See https://matplotlib.org/examples/color/colormaps_reference.html 
    for colormap names. 
    """
    
    norm = mpl.colors.Normalize(vmin=series.min(), 
                                vmax=series.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    if(dots):
        m_arr = m.to_rgba(series).reshape(len(series),4) * 180
    else:
        m_arr = m.to_rgba(series).reshape(len(series),4) * 255
    m_arr[:,3] = np.repeat(alpha, len(series))
    return list(m_arr)

def generateVoronoiPoly(filename = "siswa"):
    # Change to preferred col & index names
    sekolahs = pd.read_csv("data/" + filename + "_centroids.csv")
    
    try:
        sekolahs = sekolahs.drop(["kode"], axis=1)
    except:
        pass
    
    sekolahs[["lat","lng"]] = sekolahs[["lat","lng"]].astype(float)
    sekolahs[["jml"]] = sekolahs[["jml"]].astype(int)

    sekolahs["polygons"] = calc_polygons(sekolahs)

    sekolahs["colors"] = nums_to_color(sekolahs["jml"], alpha=0.9)

    with open("data/" + filename + "_centroids.csv", 'r') as read_obj, \
        open("data/" + filename + "_centroids_voronoi.csv", 'w', newline='') as write_obj:
            # Create a csv.reader object from the input file object
            csv_reader = reader(read_obj)
            # Create a csv.writer object from the output file object
            csv_writer = writer(write_obj)
            # Read each row of the input csv file as list
            i = 1
            for row in csv_reader:
                # Append the default text in the row / list
                if(i == 1):
                    row.append('polygons')
                    row.append('colors')
                else:
                    data = []
                    for polygon in sekolahs["polygons"][i-2]:
                        data.append(polygon)
                    row.append(data)

                    data = []
                    for polygon in sekolahs["colors"][i-2]:
                        data.append(polygon)
                    row.append(data)
                i+=1
                # Add the updated row / list to the output file
                csv_writer.writerow(row)
    write_obj.close()

    # cleaning var
    text = open("data/" + filename + "_centroids_voronoi.csv", "r")
    text = ''.join([i for i in text]) \
        .replace("array(", "")
    out = open("data/" + filename + "_centroids_voronoiclean.csv","w")
    out.writelines(text)
    out.close()
    
    text = open("data/" + filename + "_centroids_voronoiclean.csv", "r")
    text = ''.join([i for i in text]) \
        .replace(")", "")
    out = open("data/" + filename + "_centroids_voronoi.csv","w")
    out.writelines(text)
    out.close()

    os.remove("data/" + filename + "_centroids_voronoiclean.csv")

def generateClusters(filename = "siswa"):
    siswas = pd.read_csv("data/" + filename + ".csv")
    try:
        siswas = siswas.drop(["nik","lokasi","latsmp","lngsmp","jalur","jarak"], axis=1)
    except:
        siswas = siswas.drop(["nik","lokasi"], axis=1)

    kmeans = KMeans(n_clusters = 63, random_state = 1104)

    kmeans.fit(siswas)

    labels = kmeans.predict(siswas)
    centroids = kmeans.cluster_centers_

    df_processed = siswas.copy()
    df_processed['class'] = pd.Series(labels, index=df_processed.index)
    df_processed['colors'] = nums_to_color(df_processed["class"], dots=True, alpha=0.9)
    
    output = open('data/' + filename + '_clustered.csv', 'w')
    print('lat,lng,class,colors', file=output)

    for index, row in df_processed.iterrows():
        data = []
        for polygon in row["colors"]:
            data.append(polygon)

        print('{},{},{},"{}"'.format(row['lat'], row['lng'], row['class'], data), file=output)

    try:
        siswas['colors'] = nums_to_color(siswas["diterima"], dots=True, alpha=0.9)

        output = open('data/' + filename + '_colored.csv', 'w')
        print('lat,lng,class,color', file=output)

        for index, row in siswas.iterrows():
            data = []
            for polygon in row["colors"]:
                data.append(polygon)

            print('{},{},{},"{}"'.format(row['lat'], row['lng'], row['diterima'], data), file=output)
    except:
        pass
    
    classified = df_processed['class'].to_numpy()

    unique, counts = np.unique(classified, return_counts=True)
    counts = dict(zip(unique, counts))

    csv = open("data/" + filename + "_centroids.csv", 'w')
    print('kode,lat,lng,jml', file=csv)

    idx = 0
    for centroid in centroids:
        print('{},{},{},{}'.format('SMP'+str(idx+1), centroid[0], centroid[1], counts[idx]), file=csv)
        idx-=-1

def calculateDistance(lat, newlat, lng, newlng):
    R = 6373.0

    lat1 = radians(float(lat))
    lon1 = radians(float(lng))
    lat2 = radians(float(newlat))
    lon2 = radians(float(newlng))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c * 1000

    return distance

def mapping(filename = "siswa"):
    df = pd.read_csv("data/" + filename + "_centroids.csv")
    sekolah = pd.read_csv("data/sekolah.csv")

    # flag = [0] * 63
    maps = []
    for row in range(df.shape[0]):
        # kode = df.iat[row, 0]
        lat = df.iat[row, 1]
        lng = df.iat[row, 2]
        # jml = df.iat[row, 3]

        kodesek = ""
        dist = 0
        for row2 in range(sekolah.shape[0]):
            kode2 = sekolah.iat[row2, 0]
            # jml2 = sekolah.iat[row2, 1]
            lat2 = sekolah.iat[row2, 2]
            lng2 = sekolah.iat[row2, 3]
            # radius = sekolah.iat[row2, 4]

            cur_dist = calculateDistance(lat, lat2, lng, lng2)
            if(dist == 0):
                dist = cur_dist
                kodesek = kode2
            elif(cur_dist < dist):
                dist = cur_dist
                kodesek = kode2

        # flag[int(kodesek[3:]) - 1] += 1
        maps.append(kodesek)

        indices = [j for j, x in enumerate(maps) if x == kodesek]
        if(len(indices) > 1):
            jml_ril = sekolah.iat[int(kodesek[3:]) - 1, 1]
            lat_ril = sekolah.iat[int(kodesek[3:]) - 1, 2]
            lng_ril = sekolah.iat[int(kodesek[3:]) - 1, 3]

            jml1 = df.iat[indices[0], 3]
            lat1 = df.iat[indices[0], 1]
            lng1 = df.iat[indices[0], 2]

            jml2 = df.iat[indices[1], 3]
            lat2 = df.iat[indices[1], 1]
            lng2 = df.iat[indices[1], 2]
            
            # dist1 = calculateDistance(lat_ril, lat1, lng_ril, lng1)
            # dist2 = calculateDistance(lat_ril, lat2, lng_ril, lng2)
            # if(dist1 < dist2):
            if(abs(jml_ril - jml1) > abs(jml_ril - jml2)):
                update = indices[0]
                jml_fix = jml1
                lat_fix = lat1
                lng_fix = lng1
            else:
                update = indices[1]
                jml_fix = jml2
                lat_fix = lat2
                lng_fix = lng2

            arr = []
            for row2 in range(sekolah.shape[0]):
                if(row2 == int(kodesek[3:])):
                    continue

                kode3 = sekolah.iat[row2, 0]
                jml3 = sekolah.iat[row2, 1]
                lat3 = sekolah.iat[row2, 2]
                lng3 = sekolah.iat[row2, 3]
                # radius = sekolah.iat[row2, 4]

                cur_dist = calculateDistance(lat_fix, lat3, lng_fix, lng3)
                arr.append({'kode': kode3, 'dist': cur_dist, 'jml': abs(jml3 - jml_fix)})

            arr_sorted = sorted(arr, key=lambda dct: dct['dist'])[:5]
            search = sorted(arr_sorted, key=lambda dct: dct['jml'])[0]['kode']

            maps[update] = search
            
    with open("data/" + filename + "_centroids.csv", 'r') as read_obj, \
        open("data/" + filename + "_centroids2.csv", 'w', newline='') as write_obj:
            # Create a csv.reader object from the input file object
            csv_reader = reader(read_obj)
            # Create a csv.writer object from the output file object
            csv_writer = writer(write_obj)
            # Read each row of the input csv file as list
            i = 1
            for row in csv_reader:
                # Append the default text in the row / list
                if(i != 1):
                    row[0] = maps[i-2]
                i+=1
                # Add the updated row / list to the output file
                csv_writer.writerow(row)
    write_obj.close()
    
    os.remove("data/" + filename + "_centroids.csv")
    os.rename("data/" + filename + "_centroids2.csv", "data/" + filename + "_centroids.csv")

    with open("data/" + filename + "_clustered.csv", 'r') as read_obj, \
        open("data/" + filename + "_clustered2.csv", 'w', newline='') as write_obj:
            # Create a csv.reader object from the input file object
            csv_reader = reader(read_obj)
            # Create a csv.writer object from the output file object
            csv_writer = writer(write_obj)
            # Read each row of the input csv file as list
            i = 1
            for row in csv_reader:
                # Append the default text in the row / list
                if(i != 1):
                    row[2] = maps[int(row[2])][3:]
                i+=1
                # Add the updated row / list to the output file
                csv_writer.writerow(row)
    write_obj.close()

    os.remove("data/" + filename + "_clustered.csv")
    os.rename("data/" + filename + "_clustered2.csv", "data/" + filename + "_clustered.csv")

def accuracy(filename = "siswa"):
    df = pd.read_csv("data/" + filename + "_clustered.csv")
    siswa = pd.read_csv("data/" + filename + ".csv")

    total = df.count()[0]
    total_tp = siswa.count()[0]
    for row in range(df.shape[0]):
        # kode = df.iat[row, 0]
        # try:
        #     if(int(siswa.iat[row, 8]) != int(df.iat[row, 2])):
        #         total_tp -= 1
        # except:
        if(int(siswa.iat[row, 7]) != int(df.iat[row, 2])):
            total_tp -= 1

    print("Accuracy: " + str((total_tp/total)*100) + "%")

def voronoi_dots(center = (-7.2732,112.7208)):
    clusters = pd.read_csv("data/siswa_centroids_voronoi.csv")
    siswas = pd.read_csv("data/siswa_colored.csv")

    for lat, lon, color, poly, jml, kode in clusters[["lat","lng","colors","polygons","jml","kode"]].values:
        print(kode)

        m = folium.Map(location=[*center],
                    width=686, height=686, 
                    zoom_start=12,
                    api_key='6NbtVc32EkZBkf8eXLAE')

        color = literal_eval(color)
        poly = literal_eval(poly)

        points = to_convex(np.flip(poly).tolist())
        vlayer = vector_layers.Polygon(points, 
                                       fill=True, 
                                       color="black",
                                       fill_color="rgba({}, {}, {}, {})".format(*color),
                                       weight=1)
        m.add_child(vlayer)
        
        siswabysek = siswas.loc[siswas['class'] == int(kode[3:])]
        
        dist = 0
        for lat2, lon2, classified in siswabysek[["lat","lng","class"]].values:
            clayer = vector_layers.Circle([lat2,lon2], 1, color="red")
            m.add_child(clayer)

            cur_dist = calculateDistance(lat, lat2, lon, lon2)
            if(dist < cur_dist):
                dist = cur_dist

        clayer = vector_layers.Circle([lat,lon], 2, color="black")
        m.add_child(clayer)
        
        folium.Marker((lat, lon), icon=DivIcon(
            icon_size=(.1,.1),
            icon_anchor=(6,0),
            html='<div style="font-size: 8pt; color : black">%s</div>'%str(kode[3:]),
        )).add_to(m)

        clayer = vector_layers.Circle([lat,lon], dist, color="black")
        m.add_child(clayer)

        show_map(m, "sekolah/" + kode + ".png")

def hitungJarak(filename = "siswa"):
    df = pd.read_csv("data/" + filename + "_centroids.csv")
    sekolah = pd.read_csv("data/sekolah.csv")
    csv = open("data/jarak.csv", "w")
    print("kode,jarak", file=csv)

    for row in range(df.shape[0]):
        kode = df.iat[row, 0]
        lat = df.iat[row, 1]
        lng = df.iat[row, 2]

        for row in range(sekolah.shape[0]):
            kodesek = sekolah.iat[row, 0]
            latsek = sekolah.iat[row, 2]
            lngsek = sekolah.iat[row, 3]

            if(kodesek == kode):
                dist = calculateDistance(lat, latsek, lng, lngsek)
                print(kode[3:]+","+str(dist), file=csv)
                break
            else:
                continue

def jarakBar():
    fig, ax = plt.subplots()
    data = pd.read_csv("data/jarak.csv")
    data = data.sort_values(["kode"], ascending=True)
    data_jarak = data.groupby("kode")["jarak"].mean()
    data_jarak.plot(kind="bar", title="Jarak mapping dengan smp riil",  figsize=(20,10))
    
    labels = data["kode"].to_numpy()
    ax.set_xticklabels(labels)
    plt.show()

def getPilihanSiswa(tahun="2019", filename = "siswaall", sekolahfile = "sekolah", output="pilihan"):
    data = pd.read_csv("data/" + filename + ".csv")
    data = data.drop(["nik","lokasi"], axis=1)
    
    sekolah = pd.read_csv("data/" + tahun + "/" + sekolahfile + ".csv")
    sekolah = sekolah.drop(["kode","jml"], axis=1)

    try:
        sekolah = sekolah.drop(["radius"], axis=1)
    except:
        pass
    
    tempall = []
    for row in range(data.shape[0]):
        lat = data.iat[row, 0]
        lng = data.iat[row, 1]
        
        siswa = {}
        for row2 in range(sekolah.shape[0]):
            latsmp = sekolah.iat[row2, 0]
            lngsmp = sekolah.iat[row2, 1]
            
            dist = calculateDistance(lat, latsmp, lng, lngsmp)
            try:
                if(dist <= 1000):
                    siswa["1000"] += 1
            except:
                if(dist <= 1000):
                    siswa["1000"] = 1
            try:
                if(dist <= 2000):
                    siswa["2000"] += 1  
            except:
                if(dist <= 2000):
                    siswa["2000"] = 1   
            try:
                if(dist <= 3000):
                    siswa["3000"] += 1
            except:
                if(dist <= 3000):
                    siswa["3000"] = 1
                    
        tempall.append(siswa)
        
    if(tahun == '2019'):
        csv = open("data/siswaall_" + output + ".csv", "w")
    else:
        csv = open("data/" + tahun + "/siswaall_" + output + ".csv", "w")

    print("1000,2000,3000", file=csv)
    
    row = 0
    for item in tempall:
        row += 1
        
        try:
            get = item["1000"]
        except:
            item["1000"] = 0
        
        try:
            get = item["2000"]
        except:
            item["2000"] = 0
        
        try:
            get = item["3000"]
        except:
            item["3000"] = 0
            
        print(str(item["1000"])+","+str(item["2000"])+","+str(item["3000"]), file=csv)

def lineGraph(minimum = 1, inputfile="pilihan", tahun="2019"):
    if(tahun == "2019"):
        data = pd.read_csv("data/siswaall_" + inputfile + ".csv")
    else:
        data = pd.read_csv("data/2020/siswaall_" + inputfile + ".csv")

    total = len(data)

    jmlsiswa = {1: 0, 2: 0, 3: 0}
    for row in range(data.shape[0]):
        p1 = data.iat[row, 0]
        p2 = data.iat[row, 1]
        p3 = data.iat[row, 2]

        if(p1 >= minimum):
            jmlsiswa[1] += 1
        if(p2 >= minimum):
            jmlsiswa[2] += 1
        if(p3 >= minimum):
            jmlsiswa[3] += 1
    print(jmlsiswa)
    percentages = [0]
    labels = ["0", "1000", "2000", "3000"]
    for i in range(3):
        percentages.append(jmlsiswa[i+1]/total)
    print(percentages)
    
    plt.title("Presentase siswa PPDB SMPN " + tahun + " memiliki minimal " + str(minimum) + " sekolah pilihan")
    plt.xlabel("Jarak dalam KM")
    plt.plot(percentages)
    plt.show()

def barPeminatDiterima():
    fig, ax = plt.subplots()
    data = pd.read_csv("data/2020/sekolah.csv")
    data = data.sort_values(["kode"], ascending=True)
    labels = data["kode"].to_numpy()
    data = data.drop(["kode","lat","lng","radius"], axis=1)
    data.plot(kind="bar", title="Perbandingan peminat dengan diterima",  figsize=(20,10))
    
    ax.set_xticklabels(labels)
    plt.show()

if __name__ == '__main__':
    # generateClusters("siswaall")
    # generateVoronoiPoly("2020/sekolah")
    # generatePNG("2020/sekolah")

    # hitungJarak()
    # barPeminatDiterima()
    # getPilihanSiswa("2020", "2020/siswaall", "siswaall_centroids", "pilihan2020")
    # lineGraph(1, "pilihan_riil")
    # lineGraph(2, "pilihan_riil")
    lineGraph(3, "pilihan_riil", "2020")
    
    # generateClusters("siswa")
    # mapping("siswa")
    # generateVoronoiPoly("siswa")
    # generatePNG("siswa")
    # accuracy("siswa")

    # voronoi_dots()

    # nambah 1 sekolah di tempat padat
    # ini hanya sebagai medium
    # analitik
    # di atas rata2, perlu pembenaran titik sekolah
    # salah satu dari zonasi mengurangi mobilitas
    # uji performa, kualitatif, usability evaluation, jumlah klik, mouse travel,
    # fitur2 yang sama
    
    # Tugas Akhir ok

    # setelah bab apabila berakhir di halaman genap, setelahnya tambah (Halaman ini dikosongkan)

    # waktu pengujian dihitung
    # per use case: goal, task, status
    # setelah :, - gaperlu spasi

    # Terminologi khusus, nama menu => Title Case
    # Selain itu => Sentence case

    # Perancangan masih belum diimplementasi, masih perencanaan.
    # Tampilkan ppdb tahun lalu
    # Skenario, kuisioner, p
    # Daftar pustaka => diakses pada ...

    # Tabel captionnya di atas, gambar di bawah

    # Subject, predikat => Kalimat baku

    # show_map(plot_map2(), "mappingall.png")

# # heatmap  
# center=(-7.2732,112.7208)
# m = folium.Map(location=[*center],
#                    width=686, height=686, 
#                    zoom_start=12,
#                    api_key='6NbtVc32EkZBkf8eXLAE')
# HeatMap(data=sekolahs[["lat","lng","jml"]].values, radius=8, max_zoom=13).add_to(m)
# show_map(m, "heatmap.png")

# cek jarak antar cluster kmeans dengan sekolah asli berdasarkan mapping

# for i in range(1,63):
#     show_map(plot_map2(sekolahs, siswas, clusters, i), "map/mapping" + str(i) + ".png")