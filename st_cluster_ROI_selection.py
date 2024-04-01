from os import environ
N_THREADS = '64'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yaml
import bokeh
import streamlit as st
import os
import subprocess
import webbrowser
from anndata import AnnData
import scanpy as sc
# import squidpy as sq
import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import scimap as sm
import pickle
import platform, os
from pathlib import Path
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
#import dash_bio
from umap import UMAP
import glob
import os
from pdf2image import convert_from_path
from utils import show_cluster_heatmap,rename,get_heatmapdf
import os


#from streamlit_extras.stateful_button import button

#cd /Users/mengxi/Documents/utsw/codex/codex_hub/codex/
#from codex_lib.plot.exploreClusters import generate_cluster_plots, plot_clusters

st.set_option('deprecation.showPyplotGlobalUse', False)


######################################################################
st.sidebar.markdown('''
# Sections
- [Data review](#dv)
- [Tier 1 cluster](#Tier1)
- [New Tier cluster](#Tier2)

''', unsafe_allow_html=True)


# i means the clustered subset index




##########################################################################



st.title("Step wised cluster exploration of TME")
if st.button("user guide and principle"):
    """
    User guide : \n
        1. if cell number>10000, Only 10% of the data are included for cluster and visualization purposes, unless check "run whole dataset" after the clustering scheme is determined.
        2. Due to the nature of Streamlit, any modification to an option will result in the rerun. Therefore, after a complete clustering is finished, please save the YAML information and label column.
        3. Uncheck unnecessary visualizations to speed up response time on each rerun.
        4. The cluster map is interactive. Click on the legend, and can be saved locally.
        5. If you want to restart or restore the dataset, please press the button in the upper right corner to clear cache.
        6. While running in "whole dataset" mode the steps with :mantelpiece_clock: will take extra time(about 2 mins each), avoid using them while you are working with full dataset and no need to worry those functions are totally covered by others, if you insist on whole dataset please be patient.
    principle: \n

        1. increasing cell-type granularity led to decreased labeling accuracy; 
        therefore, subtle phenotype annotations should be avoided at the clustering step. 

        2. accuracy in cell-type identification varied more with normalization choice than with clustering algorithm. 

        3. unsupervised clustering better accounted for segmentation noise during cell-type annotation than hand-gating. 

        4. Z-score normalization was generally effective in mitigating the effects of noise from single-cell multiplexed imaging.
    """


output_dir = r'/Volumes/spatial-core/K/analysis/conzen/TNBC/interactive tool'#/shared/CODEX/aj/analysis/INNATE/P7T1/tissUUmap_tifs/label_masks-leiden_25_testing/Users/andrewj/analysis/codex/INNATE' #annData_clustleiden_25_testing-20221027-1940.pkl')
# Josh's annotated csv/pkl
#Josh_pkl = r'/shared/CODEX/aj/analysis/INNATE/P7T1/tissUUmap_tifs/label_masks-leiden_25_testing/annData_clustleiden_25_testing-20221027-1940.pkl'
pkl = r'/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/conzen/TNBC_Jan17/anndata/adata_normed_hamornized_louvain.pkl'
#/shared/spatial-core/K/analysis/conzen/TNBC/TNBC_ROI_1A/Stage1Pipeline_hyperion_test/20230317-1542/Stage2Pipeline_clustering_k/multi_ROI20230404-1651/z-score_0/20230404-1655/annData_clust_multi_ROIs_overall_data_clustered_20230404-1655.pkl' #annData_clustleiden_25_testing-20221027-1940.pkl')
#Josh_csv = r'/shared/CODEX/aj/analysis/INNATE/P7T1/tissUUmap_tifs/adata_df_20221027-1940_leiden_25_testing.csv' #annData_clustleiden_25_testing-20221027-1940.pkl')

pkl_ = st.file_uploader("Choose a pkl file anndata")
#csv = st.file_uploader("Choose a cvs file as cluster annotation")

@st.cache_data
def load_data(output_dir,pkl,pkl_):

    # Josh's annotated csv/pkl
    if pkl_ is not None:
        #with open(Josh_pkl, 'rb') as handle:
        b = pickle.load(pkl_).copy()
        adaClust = b.copy()
        t = b.to_df()
        
    else:
        hpc=r'/endosome/'
        mac=r'/Volumes/'
        dell=r'Z:\\'
        archive=r"archive/bioinformatics/Jamieson_lab"
        if os.path.exists(hpc):
            print("loading from bioHPC")

            path1=Path(hpc+archive+output_dir)
            path2=Path(hpc+archive+pkl)
            
        # Verify that the network drive is accessible
        elif os.path.exists(dell):
            print("loading from DELL")

            path1=Path('Z:\\'+output_dir)
            path2=Path('Z:\\'+pkl)
            
        else:
            print("loading from MAC")

            path1=Path(mac+archive+output_dir)
            # path2=Path(mac+archive+Josh_pkl)
            # path3=Path(mac+archive+Josh_csv)
            

        # output_dir = path1
        # Josh_pkl=path2
        
        with open(pkl, 'rb') as handle:
            b = pickle.load(handle).copy()
        adaClust = b.copy()
        t = b.to_df()  
        
    return b, adaClust,t

wholeset = st.checkbox("run whole dataset")
if st.checkbox("upload the pkl data"):
    b, adaClust,t=load_data(output_dir,pkl,pkl_)
    
    @st.cache_data
    def get_subsample(_b):
        st.session_state.df = sc.pp.subsample(_b, fraction=0.05, copy=True)
        return st.session_state.df
    if 'df' not in st.session_state: 
        if wholeset or t.shape[0]<10000:
            st.session_state.df = b
        else:
            st.session_state.df = get_subsample(b)



    ##########################################################################

    ##########################################################################

    st.header("dataset review",anchor="dv")

    st.write(f"shape of dataset {b.shape},column names of dataset:")
    st.write(f"shape of subsampled dataset {st.session_state.df.shape}")
    st.write("anndata info: ",adaClust)

    phenotype = st.multiselect(
        "chose one column as a 'sample' phenotype you want to compare",
        list(st.session_state.df.obs.columns),
        [list(st.session_state.df.obs.columns)[0]])[0]
    ROIregion = st.multiselect(
        "chose column of ROI region",
        list(adaClust.obs.columns),
        ['region'])[0]
    st.write(ROIregion)
    st.write(adaClust.obs['region'].unique())
    if st.checkbox("show region cell number"):
        for i in adaClust.obs['region'].unique():   
            st.write(f"In region {i}, {adaClust.obs['region'].value_counts()[i]} cells are found")
            

    
    
    plt.rcParams['figure.figsize'] = [15, 5]

    import streamlit as st

    st.markdown("""
    <style>
        .red-checkbox input[type=checkbox] + label::before {
            border: 2px solid red !important;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    

    if st.checkbox("Marker_Distributions :mantelpiece_clock:"):
        

            
        ROIregion_selection_for_maker_distribution = st.multiselect(
        "chose ROI regions",
        adaClust.obs['region'].unique(),
        adaClust.obs['region'].unique()[0]) 
        for i in ROIregion_selection_for_maker_distribution:
            figviolin = px.violin(st.session_state.df[st.session_state.df.obs["region"]==i].to_df(), box=True,
                #hover_data=st.session_state.df.to_df().columns
                )
            # sns.violinplot(data=st.session_state.df.to_df().iloc[:,:], inner="points",jitter=0.4,rotation=45).set_title('Marker Distributions')
            # figpath=r"marker_distr.png"
            # plt.savefig(figpath)
            # st.image(plt.imread(figpath))
            st.plotly_chart(figviolin)    
        
        
    if st.checkbox("cell cluster Distributions"):  
        ROIregion_selection_for_cell_cluster_distribution = st.multiselect(
        "chose ROI regions",
        adaClust.obs['region'].unique(),
        adaClust.obs['region'].unique()[0]) 
        
             
        with rc_context({'figure.figsize': (15, 10)}):
            fig=sc.pl.tsne(st.session_state.df, color=phenotype, size=20)
            plt.tight_layout()
        figpath=r"cell_distr.png"
        plt.savefig(figpath)
        st.image(plt.imread(figpath))
        for i in ROIregion_selection_for_cell_cluster_distribution:
            with rc_context({'figure.figsize': (15, 10)}):
                fig=sc.pl.tsne(st.session_state.df[st.session_state.df.obs["region"]==i], color=phenotype, size=20)
                plt.tight_layout()
            figpath="cell_distr_{}.png".format(i)
            plt.savefig(figpath)
            st.image(plt.imread(figpath))

    if st.checkbox("Marker spatial Distributions"):
        marker = st.multiselect("choose a marker to show its distribution:",
                                st.session_state.df.to_df().columns,st.session_state.df.to_df().columns[0]
                                )
        ROIregion_selection_for_Marker_spatial_distribution = st.multiselect(
        "chose ROI regions",
        adaClust.obs['region'].unique(),
        adaClust.obs['region'].unique()[0]) 
        
        plt.rcParams['figure.figsize'] = [20, 10]
        for i in ROIregion_selection_for_Marker_spatial_distribution:
            with rc_context({'figure.figsize': (15, 10)}):
                fig=sc.pl.spatial(st.session_state.df[st.session_state.df.obs["region"]==i],color=marker, spot_size=10, cmap='jet', colorbar_loc='top')
                plt.tight_layout()
        #sc.pl.spatial(adata=st.session_state.df, color=marker, spot_size=20, cmap='jet', colorbar_loc='top')
                figpath="Marker_spatial_distr_{}.png".format(i)
                plt.savefig(figpath)
                st.image(plt.imread(figpath))

    """

    """
    # data review

    st.write("all feature names",t.columns)

    all_features = list(b.var_names)
    morph = ['Area', 'MajorAxisLength', 'MinorAxisLength','Eccentricity', 'Solidity', 'Extent', 'Orientation']
    st.write("morph features",morph)
    all_markers = list(set(all_features).symmetric_difference(morph))
    st.write("markers",all_markers)







    @st.cache_data
    def glob_pdf_and_convert_to_png(path):
        pdf_files = glob.glob(os.path.join(path, '*.pdf'))
        for pdf_file in pdf_files:
            pages = convert_from_path(pdf_file)
            for page in pages:
                page.save(pdf_file.replace('.pdf', '.png'), 'PNG')



    # bf = st.session_state.df.to_df()
    # bf_t1 = AnnData(bf[marker_t1 + morph])
    # bf_t2 = AnnData(bf[marker_t2 + morph])
    # bf_t1.obsm['spatial'] = st.session_state.df.obsm['spatial']
    # bf_t2.obsm['spatial'] = st.session_state.df.obsm['spatial']
    # st.session_state.df1 = bf_t1
    # st.session_state.df2 = bf_t2




    # tsne may take 1 min
    #sc.tl.tsne(st.session_state.df)
    # st.session_state.df_t1 = sc.tl.tsne(st.session_state.df[:,marker_t1],
    #                    copy=True)
    # UMAP

    ##########################################################################


    ##########################################################################


    st.header('Tier 1 cluster',anchor="Tier1")

    if st.checkbox("start Tier 1 cluster"):

        marker_t1_ =  st.session_state.df.to_df().columns[0]
        label_t1 = ['epithelial', 'immune', 'stroma'] 
        marker_t1=st.multiselect("select Markers to use in Tier 1: ",all_markers,marker_t1_)

        method_t1=st.selectbox("what method to use in tier 1 cluster? cluster number can be assigned with kmeans",('kmeans','leiden'))#'phenograph','parc'
        # method and folow up management
        if method_t1 =='kmeans':
            k=st.number_input('Input the clusters number',min_value=2,max_value=20,value=3,step=1)
            st.write('Tier 1, will be clustered into ', k, "clusters","according to",marker_t1)
        elif method_t1=="leiden":
            k=None
            nn_ = st.slider("cluster_neighbour",min_value=10,max_value=90,value=15)
            resolution = st.slider("cluster_resolution",min_value=0.1,max_value=3.0,value=1.0)
            # @st.cache_data
            # def get_neighboured(_b,nn_):
            #     sc.pp.neighbors(_b, n_neighbors=nn_)
            # get_neighboured(st.session_state.df,nn_)
            st.write("Tier 1 cluster","according to",marker_t1)

        title1 = st.text_input('please input title for tier 1 cluster,","is not allowed', 'epithelial_immune_stroma')
        label_name_t1 = "marker_t1"+"_"+method_t1+"_"+title1
        st.write(f"your Tier 1 method {method_t1}, your title {label_name_t1}")
        if st.checkbox("confirm tier 1 cluster settings"):
        
            if k is not None:

                sm.tl.cluster(adata=st.session_state.df, method=method_t1, 
                            subset_genes=marker_t1,
                            k=k,
                            resolution=0.1, 
                            label=label_name_t1, 
                            use_raw=False)
            else:
                sm.tl.cluster(adata=st.session_state.df, method=method_t1, 
                        subset_genes=marker_t1,
                         
                        label=label_name_t1, 
                        nearest_neighbors=nn_,
                        resolution=resolution,
                        use_raw=False)
            st.write("Tier 1 cluster done")

            # show cluster heatmap
            if st.checkbox("Show Tier 1 heatmap to define the name :mantelpiece_clock:"):   
                ROIregion_selection_for_T1_heatmap = st.multiselect(
                                                                                    "chose ROI regions",
                                                                                    adaClust.obs['region'].unique(),
                                                                                    adaClust.obs['region'].unique()[0]) 
                
                for i in ROIregion_selection_for_T1_heatmap:  
                    st.write(f"----------------->> region {i} <<------------------- ")   
                    show_cluster_heatmap(st.session_state.df[st.session_state.df.obs["region"]==i],marker_t1,label_name_t1)

            if st.checkbox('View clusters composition:'):
                if st.checkbox("t-test for t1 cluster"):
                    import scipy.stats as stats
                    #ttestM=st.multiselect("select Marker: ",all_markers,marker_t1_[0])[0]
                    ttest1=st.multiselect("select cluster 1: ",np.unique(st.session_state.df.obs[label_name_t1]),np.unique(st.session_state.df.obs[label_name_t1])[0])
                    ttest2=st.multiselect("select cluster 2: ",np.unique(st.session_state.df.obs[label_name_t1]),np.unique(st.session_state.df.obs[label_name_t1])[1])
                    #@st.cache_data
                    def ttest(_df, markers,label_name_t1,ttest1,ttest2):
                        ttestdf = pd.DataFrame()
                        for i in markers:
                            ttestdf[i] = stats.ttest_ind(_df[_df.obs[label_name_t1]==ttest1[0]].to_df()[i],_df[_df.obs[label_name_t1]==ttest2[0]].to_df()[i])
                        #st.write(stats.ttest_ind(_df[_df.obs[label_name_t1]==ttest1[0]].to_df()[ttestM],_df[_df.obs[label_name_t1]==ttest2[0]].to_df()[ttestM]))
                        
                        ttestdf.index=['t-statistic','p-value']
                        return ttestdf
                    pvalue=ttest(st.session_state.df, marker_t1,label_name_t1,ttest1,ttest2)
                    st.write(pvalue)
                #@st.cache_data
                def get_heatmapdf(_df, markers,label_name_t1):
                    heatdf = pd.DataFrame(pd.concat([_df.to_df().loc[:,markers],_df.obs[label_name_t1]],axis=1).groupby(label_name_t1).mean())
                    return heatdf
                
                ROIregion_selection_for_T1_heatmatrix=st.multiselect(
                    "chose ROI regions",
                    adaClust.obs['region'].unique(),
                    adaClust.obs['region'].unique()[0])
                
                
                for i in ROIregion_selection_for_T1_heatmatrix: 
                    st.write(f"----------------->> region {i} <<------------------- ")
                    heatmatrix=get_heatmapdf(st.session_state.df[st.session_state.df.obs["region"]==i], marker_t1,label_name_t1)
                    figheatmatrix1=px.imshow(heatmatrix,text_auto=True,color_continuous_scale='temps')
                    st.plotly_chart(figheatmatrix1)
                


                    #@st.cache_data
                    def get_value_counts(_df, column):
                        value_counts_df = pd.DataFrame(_df.obs[column].value_counts()) 
                        value_counts_df.columns = [f"count"]
                        value_counts_df['percentages'] = (_df.obs[column].value_counts()/ len(_df)) * 100
                        value_counts_df['names']=value_counts_df.index
                        return value_counts_df                       
                    #countsdf0=get_value_counts(st.session_state.df,phenotype)
                    countsdf1=get_value_counts(st.session_state.df[st.session_state.df.obs["region"]==i],label_name_t1)
                    #figclusterbar0 = px.bar(countsdf0,x='names', y="count", color='names', text="count")
                    figclusterbar1 = px.bar(countsdf1,x='names', y="count", color='names', text="count")
                    #figclusterpie0 = px.pie(countsdf0,values='percentages', names='names', title='pie chart of clusters')
                    figclusterpie1 = px.pie(countsdf1,values='percentages', names='names', title='pie chart of clusters')
                    #st.plotly_chart(figclusterbar0)
                    st.plotly_chart(figclusterbar1)                
                    
                    #st.plotly_chart(figclusterpie0)
                    st.plotly_chart(figclusterpie1)


            # rename cluster results T1:
            t1_cluster_num=len(np.unique(st.session_state.df.obs[label_name_t1]))
            st.write(f" we have {t1_cluster_num} clusters in Tier 1 to make subclusters in new tiers")

            st.write(np.unique(st.session_state.df.obs[label_name_t1]))
            t1_names=st.text_input(f"please name your {t1_cluster_num} clusters,divide your names with ',' ! ")
            if st.checkbox('confirm t1 cluster name'):
                if t1_names:
                    rename(st.session_state.df, label_name_t1,label_name_t1, t1_names,t1_cluster_num,sub_cluster_group_tar=False)
                    #rename(st.session_state.df,t1_names,label_name_t1,t1_cluster_num)
                else:
                    st.warning("please Re-enter to rename")
            
                
            if st.checkbox("Show Tier 1 named cluster"):  
                ROIregion_selection_for_T1_named_cluster = st.multiselect(
                    "chose ROI regions",
                    adaClust.obs['region'].unique(),
                    adaClust.obs['region'].unique()[0])
                
                for i in ROIregion_selection_for_T1_named_cluster: 
                    st.write(f"----------------->> region {i} <<------------------- ")

                    fig1_0 = go.Figure(
                                    px.scatter(
                                        x=st.session_state.df[st.session_state.df.obs["region"]==i].obsm["X_umap"][:, 0],
                                        y=st.session_state.df[st.session_state.df.obs["region"]==i].obsm["X_umap"][:, 1],
                                        color=st.session_state.df[st.session_state.df.obs["region"]==i].obs[label_name_t1].astype(str),
                                        color_continuous_scale="viridis",
                                        #line_width=0.5,
                                        #showscale=True,
                                        ),
                                    )
                                    

                                    # Update the layout
                    fig1_0.update_layout(
                    title="cells colored by stepwised cluster",
                    xaxis_title="UMAP 1",
                    yaxis_title="UMAP 2",
                    width=800,
                    height=800,
                    )
                    fig1_1 = go.Figure(
                                    px.scatter(
                                        x=st.session_state.df[st.session_state.df.obs["region"]==i].obs["X_centroid"],
                                        y=st.session_state.df[st.session_state.df.obs["region"]==i].obs["Y_centroid"].max()-st.session_state.df[st.session_state.df.obs["region"]==i].obs["Y_centroid"],
                                        color=st.session_state.df[st.session_state.df.obs["region"]==i].obs[label_name_t1].astype(str),
                                        color_continuous_scale="viridis",
                                        #line_width=0.5,
                                        #showscale=True,
                                        ),
                                    )
                                    

                                    # Update the layout
                    fig1_1.update_layout(
                    title="cells colored by stepwised cluster",
                    xaxis_title="X",
                    yaxis_title="Y",
                    width=800,
                    height=800,
                    )

                    # fig4 = sc.pl.spatial(st.session_state.df, color=label_name_t2, spot_size=80,    
                    #                     colorbar_loc='top', title="Spatial")

                    st.plotly_chart(fig1_0)
                    st.plotly_chart(fig1_1)
                    # st.pyplot(fig1_1)


    ##########################################################################
            

    ##########################################################################

            
            if 't2key' not in st.session_state:
                st.session_state.t2key = [label_name_t1]
            if 'i2list' not in st.session_state:
                st.session_state.i2list = []
            if 'i2' not in st.session_state:
                st.session_state.i2 = None
            

            st.header("New Tier cluster",anchor="Tier2")
            st.write('all the clustered labels:',st.session_state.t2key)

            tar_label_t2=st.multiselect("select the target cluster label",st.session_state.t2key,st.session_state.t2key[-1])[0]
            st.write("target label has following cluster:",np.unique(st.session_state.df.obs[tar_label_t2]))
            
            st.session_state.i2 = st.selectbox("select the subset cluster from target cluster label : ",np.unique(st.session_state.df.obs[tar_label_t2]))
            
            if st.checkbox(f"start clustering subset {st.session_state.i2} from target cluster label"):
                
                
                    
                st.write("clustered label",st.session_state.t2key)
                st.write("clustered subset",st.session_state.i2list)
                
                sub_cluster_group_tar=st.session_state.i2
                # define maker list for cluster
                marker_t2_ =  st.session_state.df.to_df().columns[1] #'HLA-DR'
                marker_t2=st.multiselect("select Markers: ",all_markers,marker_t2_)
                set([t in list(b.var_names) for t in marker_t2]) == {True}
                remaining_markers = list(set(all_markers).symmetric_difference(set(marker_t1).union(set(marker_t2))))
                st.write("remaining_markers",remaining_markers)
                # define method for cluster
                method_t2=st.selectbox("what method to use in new tier (leiden recommonded)?",('kmeans','leiden'))
                title2 = st.text_input('cluster title', 'sub_')
                label_name_t2 =method_t2+"_sub_"+sub_cluster_group_tar+"_"+title2

                # Use Scimap to cluster
                print('Clustering New Tier with {}'.format(marker_t2))

                @st.cache_data
                def cluster_t2(_df,k2,nn_2,resolution2,tar_label_t2,marker_t2,method_t2,label_name_t2,sub_cluster_group_tar):
                    if k2 is not None:
                    
                        sm.tl.cluster(adata=_df, method='kmeans', 
                                    subset_genes=marker_t2,
                                    k=k2,
                                    sub_cluster=True,
                                    sub_cluster_column=tar_label_t2,
                                    sub_cluster_group=sub_cluster_group_tar,
                                    resolution=0.2, 
                                    label=label_name_t2,
                                    use_raw=False)
                    else:
                        sm.tl.cluster(adata=_df, method=method_t2, 
                                subset_genes=marker_t2,
                                sub_cluster=True,
                                sub_cluster_column=tar_label_t2,
                                sub_cluster_group=sub_cluster_group_tar,
                                resolution=resolution2, 
                                label=label_name_t2,
                                nearest_neighbors=nn_2,
                                use_raw=False)
                        
                if method_t2 =='kmeans':
                    k2=st.number_input('Input new tier cluster number',min_value=2,max_value=20,value=3,step=1)                                
                    st.write('new tier, will be clustered into ', k2, " clusters with method ",method_t2," according to ",marker_t2) 
                    nn_2=None
                    resolution2=None                  
                elif method_t2=="leiden":
                    k2=None
                    nn_2 = st.slider("cluster_neighbour num for new tier",min_value=10,max_value=90,value=15)
                    resolution2 = st.slider("cluster_resolution for new tier",min_value=0.1,max_value=3.0,value=1.0)
                    # @st.cache_data
                    # def get_neighboured(_b,nn_):
                    #     sc.pp.neighbors(_b, n_neighbors=nn_)
                    # get_neighboured(st.session_state.df,nn_)                
                    st.write("new tier, will be clustered into ", nn_2, " clusters with method ",method_t2," according to ",marker_t2)
                        
                #st.stop()          

                if st.checkbox("confirm new tier cluster settings"): 
                    cluster_t2(st.session_state.df,k2,nn_2,resolution2,tar_label_t2,marker_t2,method_t2,label_name_t2,sub_cluster_group_tar)

                    bsub=st.session_state.df[st.session_state.df.obs[tar_label_t2]==sub_cluster_group_tar, marker_t2]
                    #st.write("bsub info",bsub)
                    st.write("st.session_state.df info",st.session_state.df)
                    
                                        # show cluster heatmap
                    if st.checkbox("Show new tier heatmap to define the name :mantelpiece_clock:"): 
                        ROIregion_selection_for_new_tier_heatmap = st.multiselect(
                                                                                    "chose ROI regions",
                                                                                    adaClust.obs['region'].unique(),
                                                                                    adaClust.obs['region'].unique()[0])     
                        for i in ROIregion_selection_for_new_tier_heatmap:  
                            st.write(f"----------------->> region {i} <<-------------------")   
                            show_cluster_heatmap(bsub[bsub.obs["region"]==i],marker_t2,label_name_t2)          
                        #show_cluster_heatmap(bsub,marker_t2,label_name_t2)

                    if st.checkbox('View new tier clusters composition:'):

                        if st.checkbox("t-test for new cluster"):
                            import scipy.stats as stats
                            #ttestM=st.multiselect("select Marker: ",all_markers,marker_t1_[0])[0]
                            ttest2_1=st.multiselect("select new cluster 1: ",np.unique(st.session_state.df.obs[label_name_t2]),np.unique(st.session_state.df.obs[label_name_t2])[0])
                            ttest2_2=st.multiselect("select new cluster 2: ",np.unique(st.session_state.df.obs[label_name_t2]),np.unique(st.session_state.df.obs[label_name_t2])[1])
                            #@st.cache_data
                            def ttest(_df, markers,label_name_t1,ttest1,ttest2):
                                ttestdf = pd.DataFrame()
                                for i in markers:
                                    ttestdf[i] = stats.ttest_ind(_df[_df.obs[label_name_t1]==ttest1[0]].to_df()[i],_df[_df.obs[label_name_t1]==ttest2[0]].to_df()[i])
                                #st.write(stats.ttest_ind(_df[_df.obs[label_name_t1]==ttest1[0]].to_df()[ttestM],_df[_df.obs[label_name_t1]==ttest2[0]].to_df()[ttestM]))
                                return ttestdf
                            pvalue2=ttest(st.session_state.df, marker_t2,label_name_t2,ttest2_1,ttest2_2)
                            st.write(pvalue2)
                        # @st.cache_data
                        # def get_heatmapdf(_df, markers,label_name):
                        #     heatdf = pd.DataFrame(pd.concat([_df.to_df().loc[:,markers],_df.obs[label_name]],axis=1).groupby(label_name).mean())
                        #     st.write(heatdf)
                        #     return heatdf
                        #==========================================
                        ROIregion_selection_for_new_tier_heatmapmatrix=st.multiselect(
                            "chose ROI regions",
                            adaClust.obs['region'].unique(),
                            adaClust.obs['region'].unique()[0])
                        
                        for i in ROIregion_selection_for_new_tier_heatmapmatrix: 
                            st.write(f"----------------->> region {i} <<------------------- ")
                            heatmatrix=get_heatmapdf(bsub[bsub.obs["region"]==i], marker_t2,label_name_t2)
                            figheatmatrix1=px.imshow(heatmatrix,text_auto=True,color_continuous_scale='temps')
                            st.plotly_chart(figheatmatrix1)
                        
                            def get_value_counts(_df, column):
                                value_counts_df = pd.DataFrame(_df.obs[column].value_counts()) 
                                value_counts_df.columns = [f"count"]
                                value_counts_df['percentages'] = (_df.obs[column].value_counts()/ len(_df)) * 100
                                value_counts_df['names']=value_counts_df.index #np.unique(_df.obs[column])
                                #st.write(value_counts_df['names'])
                                return value_counts_df  

                            countsdf0=get_value_counts(st.session_state.df[st.session_state.df.obs["region"]==i],phenotype)
                            countsdf1=get_value_counts(st.session_state.df[st.session_state.df.obs["region"]==i],label_name_t2)
                            figclusterbar0 = px.bar(countsdf0,x='names', y="count", color='names', text="count")
                            figclusterbar1 = px.bar(countsdf1,x='names', y="count", color='names', text="count")
                            figclusterpie0 = px.pie(countsdf0,values='percentages', names='names', title='pie chart of clusters')
                            figclusterpie1 = px.pie(countsdf1,values='percentages', names='names', title='pie chart of clusters')
                            st.plotly_chart(figclusterbar0)
                            st.plotly_chart(figclusterbar1)
                            st.plotly_chart(figclusterpie0)
                            st.plotly_chart(figclusterpie1)

                        #==========================================
                        # heatmatrix2=get_heatmapdf(bsub, marker_t2,label_name_t2)
                        # figheatmatrix2=px.imshow(heatmatrix2,text_auto=True,color_continuous_scale='temps')
                        # st.plotly_chart(figheatmatrix2)

                        # #@st.cache_data
                        # def get_value_counts(_df, column):
                        #     value_counts_df = pd.DataFrame(_df.obs[column].value_counts()) 
                        #     value_counts_df.columns = [f"count"]
                        #     value_counts_df['percentages'] = (_df.obs[column].value_counts()/ len(_df)) * 100
                        #     value_counts_df['names']=value_counts_df.index #np.unique(_df.obs[column])
                        #     st.write(value_counts_df['names'])
                        #     return value_counts_df                       
                        # countsdf0=get_value_counts(st.session_state.df,phenotype)
                        # countsdf1=get_value_counts(st.session_state.df,label_name_t2)
                        # figclusterbar0 = px.bar(countsdf0,x='names', y="count", color='names', text="count")
                        # figclusterbar1 = px.bar(countsdf1,x='names', y="count", color='names', text="count")
                        # figclusterpie0 = px.pie(countsdf0,values='percentages', names='names', title='pie chart of clusters')
                        # figclusterpie1 = px.pie(countsdf1,values='percentages', names='names', title='pie chart of clusters')
                        # st.plotly_chart(figclusterbar0)
                        # st.plotly_chart(figclusterbar1)                
                        
                        # st.plotly_chart(figclusterpie0)
                        # st.plotly_chart(figclusterpie1)


                    
                    t2_cluster_num=len(np.unique(bsub.obs[label_name_t2]))
                    st.write(f" we have {np.unique(st.session_state.df.obs[st.session_state.t2key[-1]])} clusters in previous Tier to make subclusters in new Tier")
                    st.write(f" we choose {st.session_state.i2} cluster in last round Tier to subcluster in to {t2_cluster_num} clusters")
                    
                    # rename cluster results T1:
                    t2_names=st.text_input(f"please name your {t2_cluster_num} clusters in cluster {st.session_state.i2}, devide your names with ',' ! ")
                    if st.checkbox('confirm new tier cluster name'):

                        #@st.cache_data
                        def rename(_df, tar_label_t2,label_name_t2, t2_names,sub_cluster_group_tar=False):
                            lent2=len(t2_names.split(","))
                            st.write(f"you have {lent2} names for {t2_cluster_num} clusters")
                            assert len(t2_names.split(","))==t2_cluster_num,"num of name must be the same of num cluster"
                            t2_name_list=t2_names.split(",")
                            rename={}
                            for i,j in enumerate(t2_name_list):

                                
                                if sub_cluster_group_tar:
                                    prefix = os.path.commonprefix(list(_df.obs[tar_label_t2][_df.obs[tar_label_t2]==sub_cluster_group_tar]))
                                    #st.write(prefix)

                                    suffixes = [x.replace(prefix, '') for x in _df.obs[tar_label_t2][_df.obs[tar_label_t2]==sub_cluster_group_tar].tolist()]
                                    rename[prefix+'-'+j]=prefix+'-'+str(i) 
                                else:
                                    rename[j]=str(i)
                            st.write(rename)
                            _df=sm.hl.rename(_df, rename, from_column=label_name_t2, to_column=label_name_t2) 

                        rename(st.session_state.df, tar_label_t2,label_name_t2, t2_names,sub_cluster_group_tar)

                        #rename(st.session_state.df,t2_names,label_name_t2,t2_cluster_num,sub_cluster_group_tar=sub_cluster_group_tar)
                        # st.write(np.unique(st.session_state.st.session_state.df[st.session_state.df.obs[st.session_state.t2key[-1]]==sub_cluster_group_tar, marker_t2]
                        # .obs[label_name_t2]))
                        #st.session_state.df[st.session_state.df.obs[st.session_state.t2key[-1]]==sub_cluster_group_tar, marker_t2]=st.session_state.st.session_state.df[st.session_state.df.obs[st.session_state.t2key[-1]]==sub_cluster_group_tar, marker_t2]  
                        st.write(np.unique(st.session_state.df.obs[label_name_t2]))
                            


                    ######
                        # if st.checkbox("show tier 2 cluster:"): 
                        #     name2="T2cluster"+sub_cluster_group_tar
                        #     km2=sm.pl.cluster_plots (st.session_state.df[st.session_state.df.obs[st.session_state.t2key[-1]]==sub_cluster_group_tar, marker_t2]
                        # , group_by=label_name_t2,output_dir=name2)
                        #     glob_pdf_and_convert_to_png(name2)
                        #     for png_file in glob.glob(name2 + '/*.png'):
                        #         st.image(png_file,use_column_width="never")
                        #     sp2=sc.pl.spatial(st.session_state.df[st.session_state.df.obs[st.session_state.t2key[-1]]==sub_cluster_group_tar, marker_t2]
                        # , color=label_name_t2, spot_size=100,    
                        #                         colorbar_loc='top', title=label_name_t2)
                        #     st.pyplot(sp2)


                        if st.checkbox("show tsne plot in whole dataset"):
                            ROIregion_selection_for_new_tier_tsne = st.multiselect(
                                "chose ROI regions",
                                adaClust.obs['region'].unique(),
                                adaClust.obs['region'].unique()[0])

                            for i in ROIregion_selection_for_new_tier_tsne:
                                st.write(f"----------------->> region {i} <<------------------- ")
                                fig3, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
                                sc.pl.tsne(st.session_state.df[st.session_state.df.obs["region"]==i], color=label_name_t1, size=10, 
                                                    colorbar_loc='top', title=label_name_t1,ax=ax0, show=False)

                                # sc.pl.tsne(st.session_state.df_t1, color=clust_names[-4], size=10, 
                                #                       colorbar_loc='top', title="t1_makers Only", ax=ax1, show=False)
                                sc.pl.tsne(st.session_state.df[st.session_state.df.obs["region"]==i], color=label_name_t2, size=5, 
                                                    colorbar_loc='top', title=label_name_t2, ax=ax1, show=False)

                                # Show the plot
                                st.pyplot(fig3)
                                fig3_1 = sc.pl.spatial(st.session_state.df[st.session_state.df.obs["region"]==i], color=tar_label_t2, spot_size=5,    
                                colorbar_loc='top', title="Spatial")
                                fig3_2 = sc.pl.spatial(st.session_state.df[st.session_state.df.obs["region"]==i], color=label_name_t2, spot_size=8,    
                                                    colorbar_loc='top', title="Spatial")
                                #st.pyplot(fig3_1)
                                st.pyplot(fig3_2)


                            
                                #if len(list(set(np.unique(st.session_state.df.obs[label_name_t1])).difference(set(np.unique(st.session_state.2ikey)))))==0:

                        if st.checkbox("check and compare new Tier results"):
                            #st.write(st.session_state.df.obs)
                            import plotly.express as px
                            ROIregion_selection_for_new_tier_results = st.multiselect(
                                "chose ROI regions",
                                adaClust.obs['region'].unique(),
                                adaClust.obs['region'].unique()[0])
                            
                            for i in OIregion_selection_for_new_tier_results:
                                st.write(f"----------------->> region {i} <<------------------- ")
                                fig00 = go.Figure(
                                                px.scatter(
                                                    x=st.session_state.df[st.session_state.df.obs["region"]==i].obsm["X_umap"][:, 0],
                                                    y=st.session_state.df[st.session_state.df.obs["region"]==i].obsm["X_umap"][:, 1],
                                                    color=st.session_state.df[st.session_state.df.obs["region"]==i].obs[label_name_t2].astype(str),
                                                    color_continuous_scale="viridis",
                                                    #line_width=0.5,
                                                    #showscale=True,
                                                    ),
                                                )
                                                

                                                # Update the layout
                                fig00.update_layout(
                                title="cells colored by stepwised cluster",
                                xaxis_title="UMAP 1",
                                yaxis_title="UMAP 2",
                                width=800,
                                height=800,
                                )

                                # fig4 = sc.pl.spatial(st.session_state.df, color=label_name_t2, spot_size=80,    
                                #                     colorbar_loc='top', title="Spatial")

                                st.plotly_chart(fig00)
                                #st.pyplot(fig4)
                                fig0 = go.Figure(
                                        px.scatter(
                                            x=st.session_state.df[st.session_state.df.obs["region"]==i].obsm["X_umap"][:, 0],
                                            y=st.session_state.df[st.session_state.df.obs["region"]==i].obsm["X_umap"][:, 1],
                                            color=st.session_state.df[st.session_state.df.obs["region"]==i].obs[phenotype].astype(str),
                                            color_continuous_scale="viridis",
                                            #line_width=0.5,
                                            #showscale=True,
                                            ),
                                        )
                                                

                                                # Update the layout
                                fig0.update_layout(
                                title="cells colored by phenotype",
                                xaxis_title="UMAP 1",
                                yaxis_title="UMAP 2",
                                width=800,
                                height=800,
                                )

                                # fig4 = sc.pl.spatial(st.session_state.df, color=label_name_t2, spot_size=80,    
                                #                     colorbar_loc='top', title="Spatial")

                                st.plotly_chart(fig0)

                                fig04 = go.Figure(
                                        px.scatter(
                                            x=st.session_state.df[st.session_state.df.obs["region"]==i].obs["X_centroid"],
                                            y=st.session_state.df[st.session_state.df.obs["region"]==i].obs["Y_centroid"].max()-st.session_state.df[st.session_state.df.obs["region"]==i].obs["Y_centroid"],
                                            color=st.session_state.df[st.session_state.df.obs["region"]==i].obs[label_name_t2].astype(str),
                                            color_continuous_scale="viridis",
                                            #line_width=0.5,
                                            #showscale=True,
                                            ),
                                        )
                                                

                                                # Update the layout
                                fig04.update_layout(
                                title="cells colored by clusters",
                                xaxis_title="X",
                                yaxis_title="Y",
                                width=800,
                                height=800,
                                )

                                # fig4 = sc.pl.spatial(st.session_state.df, color=label_name_t2, spot_size=80,    
                                #                     colorbar_loc='top', title="Spatial")

                                st.plotly_chart(fig04)

                                fig05 = go.Figure(
                                        px.scatter(
                                            x=st.session_state.df[st.session_state.df.obs["region"]==i].obs["X_centroid"],
                                            y=st.session_state.df[st.session_state.df.obs["region"]==i].obs["Y_centroid"].max()-st.session_state.df[st.session_state.df.obs["region"]==i].obs["Y_centroid"],
                                            color=st.session_state.df[st.session_state.df.obs["region"]==i].obs[phenotype].astype(str),
                                            color_continuous_scale="viridis",
                                            #line_width=0.5,
                                            #showscale=True,
                                            ),
                                        )
                                                

                                                # Update the layout
                                fig05.update_layout(
                                title="cells colored by phenotype",
                                xaxis_title="x",
                                yaxis_title="y",
                                width=800,
                                height=800,
                                )

                                # fig4 = sc.pl.spatial(st.session_state.df, color=label_name_t2, spot_size=80,    
                                #                     colorbar_loc='top', title="Spatial")

                                st.plotly_chart(fig05)


                        if st.checkbox("Save clustered df as csv/pkl"):
                            st.write(st.session_state.df.obs)
                            dftitle=st.text_input("input the file name","2-steps_clustered_")
                            # if st.button("save pkl data"):            
                            #     with open(dftitle+".pkl",'wb') as f :
                            #         pickle.dump(st.session_state.df,f)

                            @st.cache_data
                            def convert_df(_df):
                                # IMPORTANT: Cache the conversion to prevent computation on every rerun

                                # Convert AnnData object to DataFrame
                                df = _df.obs

                                # Write DataFrame to CSV file(s)
                                #df.to_csv('mydata.csv')

                                return df.to_csv()

                            csv = convert_df(st.session_state.df)   
                            
                            st.download_button(
                                label="Download clustering info as CSV",
                                data=csv,
                                file_name=dftitle+".csv",
                                mime='text/csv',
                            )
                            
                            # st.download_button(
                            #     label="Download data as pkl",
                            #     data=st.session_state.df,
                            #     file_name=dftitle+".pkl",
                            #     mime='application/octet-stream',
                            # )
                            if st.button("save pkl data"):            
                                with open(dftitle+".pkl",'wb') as f :
                                    pickle.dump(st.session_state.df,f)
                        
                    
                    if st.checkbox("review and save info to yaml start next run"):

                        cluster_dict={"col name of label T1": label_name_t1,
                                    "clusters num in T1": len(np.unique(st.session_state.df.obs[label_name_t1])),
                                    "chosen makers for T1": marker_t1,
                                    "names of T1 clusters": t1_names,
                                    
                                    "previous Tier label": st.session_state.t2key[-1],
                                    "new Tier label": label_name_t2,
                                    "clusters from previous Tier": np.unique(st.session_state.df.obs[st.session_state.t2key[-1]]),
                                    "clusters num in current round label": len(np.unique(st.session_state.df.obs[label_name_t2])),
                                    
                                    "the chosen target subcluster in new Tier": st.session_state.i2,
                                    "how many sub-subclusters were generated from target subcluster": t2_cluster_num,
                                    #"clusters in new Tier": np.unique(st.session_state.df.obs[label_name_t2]),
                                    "names of new clusters": t2_names,
                                    "chosen markers for new Tier": marker_t2,
                                    #"names of T2 clusters": st.session_state.i2+label_name_t2
                                    }
                        st.write("New Tier info: ",cluster_dict)
                        yaml_filename = st.text_input("yaml file name","Cluster_param.yaml")
                        if st.checkbox("save yaml"):
                            with open(yaml_filename, "w") as f:
                                yaml.dump(cluster_dict, f)#, default_flow_style=False, allow_unicode=True
                            
                            if st.checkbox("save label"):

                                st.session_state.t2key.append(label_name_t2)
                                st.session_state.i2list.append(st.session_state.i2)
                                st.write("clustered labels",st.session_state.t2key)
                                st.write("analysied subclusters",st.session_state.i2list)
                                #st.stop()

                            st.markdown('''
                            ### Jump to
                            - [start a new Tier](#Tier2)
                            ''', unsafe_allow_html=True)



                        

                













