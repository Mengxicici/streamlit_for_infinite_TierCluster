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
import dash_bio
from umap import UMAP
import glob
import os
from pdf2image import convert_from_path

def get_spatial(b1sub,label_name_t1):
    
    sp1=sc.pl.spatial(b1sub, color=label_name_t1, spot_size=10,    
            colorbar_loc='top', title=label_name_t1)

def glob_pdf_and_convert_to_png(path):
    pdf_files = glob.glob(os.path.join(path, '*.pdf'))
    for pdf_file in pdf_files:
        pages = convert_from_path(pdf_file)
        for page in pages:
            page.save(pdf_file.replace('.pdf', '.png'), 'PNG')


def show_cluster_heatmap(df,marker_t1,label_name_t1):

    ########################################################################
    # failed try to use dash_bio
    # fig1_0_1 = go.Figure(data=go.Heatmap(
    #                 z=st.session_state.df.to_df(),
    #                 x=st.session_state.df.var_names,
    #                 y=st.session_state.df.obs[label_name_t1],
    #                 hoverongaps = False))
    # fig1_0_2 = dash_bio.Clustergram(
    #                     data=st.session_state.df.to_df()[marker_t1],
    #                     column_labels=list(st.session_state.df.var_names),
    #                     row_labels=list(st.session_state.df.obs[label_name_t1]),
    #                     height=800,
    #                     width=700,
    # )

    # fig1_0_3 = go.Figure(data=go.Heatmap(
    #     z=st.session_state.df.to_df()[marker_t1],
    #     x=st.session_state.df.var_names,
    #     y=st.session_state.df.obs[label_name_t1].astype(str),
    #     hoverongaps = False))
    # st.plotly_chart(fig1_0_1)
    # st.plotly_chart(fig1_0_2)

    ####################################################################
    name1="T1cluster"
    name1_1="T1cluster1"
    #fig1=sc.pl.clustermap(df, group_by=label_name_t1, show=False)
    b1sub=df[:, marker_t1]
    fig1=sm.pl.cluster_plots(b1sub, group_by=label_name_t1,output_dir=name1,size=20)
    #fig1_1=sm.pl.cluster_plots(df, group_by=label_name_t1,output_dir=name1)
    # Open the PDF document
    glob_pdf_and_convert_to_png(name1)
    st.image(name1+'/_matrixplot.png',use_column_width="never")
    # for png_file in glob.glob(name1 + '/*.png'):
    #     st.image(png_file,use_column_width="never")
    st.pyplot(get_spatial(b1sub,label_name_t1))

    features = df.to_df()#[marker_t1]

    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    #umap_3d = UMAP(n_components=3, init='random', random_state=0)

    proj_2d = umap_2d.fit_transform(features)
    #proj_3d = umap_3d.fit_transform(features)

    fig_2d = px.scatter(
        proj_2d, x=0, y=1,
        color=df.obs[label_name_t1], labels={'color': 'label_name_t1'},size_max=10
    )
    # fig_3d = px.scatter_3d(
    #     proj_3d, x=0, y=1, z=2,
    #     color=df.obs[label_name_t1], labels={'color': 'label_name_t1'}
    # )
    # fig_3d.update_traces(marker_size=5)
    st.plotly_chart(fig_2d)
    #st.plotly_chart(fig_3d)


def rename(_df, tar_label,label_name_t2, t2_names,t2_cluster_num,sub_cluster_group_tar=False):
    lent2=len(t2_names.split(","))
    st.write(f"you have {lent2} names for {t2_cluster_num} clusters")
    assert len(t2_names.split(","))==t2_cluster_num,"num of name must be the same of num cluster"
    t2_name_list=t2_names.split(",")
    rename={}
    for i,j in enumerate(t2_name_list):

        
        if sub_cluster_group_tar:
            prefix = os.path.commonprefix(list(_df.obs[tar_label][_df.obs[tar_label]==sub_cluster_group_tar]))
            st.write(prefix)

            #suffixes = [x.replace(prefix, '') for x in _df.obs[tar_label][_df.obs[tar_label]==sub_cluster_group_tar].tolist()]
            rename[prefix+'-'+j]=prefix+'-'+str(i) 
        else:
            rename[j]=str(i)
    st.write(rename)
    _df=sm.hl.rename(_df, rename, from_column=label_name_t2, to_column=label_name_t2) 



#     lent2=len(t2_names.split(","))
# st.write(f"you have {lent2} names for {t2_cluster_num} clusters")
# assert len(t2_names.split(","))==t2_cluster_num,"num of name must be the same of num cluster"
# t2_name_list=t2_names.split(",")
# rename={}
# for i,j in enumerate(t2_name_list):

    
#     if sub_cluster_group_tar:
#         prefix = os.path.commonprefix(list(st.session_state.df.obs[st.session_state.t2key[-1]][st.session_state.df.obs[st.session_state.t2key[-1]]==sub_cluster_group_tar]))
#         st.write(prefix)

#         suffixes = [x.replace(prefix, '') for x in st.session_state.df.obs[st.session_state.t2key[-1]][st.session_state.df.obs[st.session_state.t2key[-1]]==sub_cluster_group_tar].tolist()]
#         rename[prefix+'-'+j]=prefix+'-'+str(i) 
#     else:
#         rename[j]=str(i)
# st.write(rename)
# st.session_state.df=sm.hl.rename(st.session_state.df, rename, from_column=label_name_t2, to_column=label_name_t2) 


@st.cache_data
def get_heatmapdf(_df, markers,label_name):
    heatdf = pd.DataFrame(pd.concat([_df.to_df().loc[:,markers],_df.obs[label_name]],axis=1).groupby(label_name).mean())
    #st.write(heatdf)
    return heatdf