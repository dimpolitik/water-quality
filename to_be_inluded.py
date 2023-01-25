# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 23:09:42 2022

@author: ΔΗΜΗΤΡΗΣ
"""


def greek_map():
    sf = shp.Reader('./coastline/coastLine.shp')        
    fig = plt.figure(figsize = (4,4))  
    for shape in sf.shapeRecords():       
        # end index of each components of map
        l = shape.shape.parts          
        len_l = len(l)  # how many parts of countries i.e. land and islands
        x = [i[0] for i in shape.shape.points[:]] # list of latitude
        y = [i[1] for i in shape.shape.points[:]] # list of longitude
                    
        l.append(len(x)) # ensure the closure of the last component
        for k in range(len_l):
            # draw each component of map.
            # l[k] to l[k + 1] is the range of points that make this component
            plt.plot(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'k-') 
            plt.fill(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'black', alpha=0.3)         
    plt.xlim([19.25, 28.6])
    plt.ylim([34.5, 42])
    plt.axis('off')
    #plt.title('River site location with ecological status')
    plt.savefig('greece.jpg', bbox_inches='tight', dpi = 600)

greek_map()


if response not in '-':   
    if (bg or bm or bp):   
        v = instance.iloc[:,:-1].values
        v  = np.array(v.flatten())
        #v = [int(x) if type(x) == int else np.round(x,3) for x in v]
        default_vars = pd.DataFrame(v, index = var[:-1], columns = ['Default values'])
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = plt.figure(figsize = (4,4))  
            sf = shp.Reader('./coastline/coastLine.shp')        
            for shape in sf.shapeRecords():       
                # end index of each components of map
                l = shape.shape.parts          
                len_l = len(l)  # how many parts of countries i.e. land and islands
                x = [i[0] for i in shape.shape.points[:]] # list of latitude
                y = [i[1] for i in shape.shape.points[:]] # list of longitude
                        
                l.append(len(x)) # ensure the closure of the last component
                for k in range(len_l):
                    # draw each component of map.
                    # l[k] to l[k + 1] is the range of points that make this component
                    plt.plot(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'k-') 
                    plt.fill(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'black', alpha=0.3)         
            plt.xlim([19.25, 28.6])
            plt.ylim([34.5, 42])
            plt.title('River site location with ecological status')
        
            if bg: ecol_class = classes[0]
            if bm: ecol_class = classes[1]
            if bp: ecol_class = classes[2]
        
            plt.scatter(dfs['Lon'], dfs['Lat'], s=25)
            plt.text(dfs['Lon']-0.2, dfs['Lat']+0.1, ecol_class, fontsize = 12, color = "red")
            
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)
        with col2:        
            st.table(default_vars)
        
        with col3:
            if w4:
                scenario_exp = np.expand_dims(scenario, axis=0)
                pred = model.predict(scenario_exp)
                        
                if (pred == 0): pred = 'High/Good'
                if (pred == 1): pred = 'Moderate'  
                if (pred == 2): pred = 'Poor/Bad'
                     
                st.write('New prediction is: ', pred)
               
         
