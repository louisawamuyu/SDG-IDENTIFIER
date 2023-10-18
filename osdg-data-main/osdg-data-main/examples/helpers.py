# standard library
from _typeshed import IdentityFunction
from codecs import getincrementaldecoder
from ctypes.wintypes import SERVICE_STATUS_HANDLE
from distutils.util import change_root
from doctest import REPORTING_FLAGS
from http.client import PAYMENT_REQUIRED
from importlib import resources
from msilib import Control
from multiprocessing import reduction
from os import supports_bytes_environ, system
from socket import J1939_MAX_UNICAST_ADDR
from tkinter.ttk import Progressbar
from token import RIGHTSHIFT
from typing import Collection, List, dataclass_transform
from textwrap import wrap
from urllib.request import AbstractBasicAuthHandler

from pandas.core.arraylike import maybe_dispatch_ufunc_to_dunder_op
from sklearn.exceptions import EfficiencyWarning

# data wrangling
import numpy as np
import pandas as pd

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# data modelling
from sklearn.metrics import confusion_matrix, accuracy_score, coverage_error, f1_score

# other settings
sns.set(
    style = 'whitegrid',
    palette = 'tab10',
    font_scale = 1.5,
    rc = {
        'figure.figsize': (12, 5),
        'axes.labelsize': 16
    }
)

def plot_confusion_matrix(y_true: np.ndarray, y_hat: np.ndarray, figsize = (16, 9)):
    """
    Convenience function to display a confusion matrix in a graph.
    """
    labels = sorted(list(set(y_true)))
    df_lambda = pd.DataFrame(
        confusion_matrix(y_true, y_hat),
        index = labels,
        columns = labels
    )
    acc = accuracy_score(y_true, y_hat)
    f1s = f1_score(y_true, y_hat, average = 'weighted')

    fig, ax = plt.subplots(figsize = figsize)
    sns.heatmap(
        df_lambda, annot = True, square = True, cbar = False,
        fmt = 'd', linewidths = .5, cmap = 'YlGnBu',
        ax = ax
    )
    ax.set(
        title = f'Accuracy: {acc:.2f}, F1 (weighted): {f1s:.2f}',
        xlabel = 'Predicted',
        ylabel = 'Actual'
    )
    fig.suptitle('Confusion Matrix')
    plt.tight_layout()

def get_top_features(vectoriser, clf, selector = None, top_n: int = 25, how: str = 'long'):
    """
    Convenience function to extract top_n predictor per class from a model.
    """

    assert hasattr(vectoriser, 'get_feature_names')
    assert hasattr(clf, 'coef_')
    assert hasattr(selector, 'get_support')
    assert how in {'long', 'wide'}, f'how must be either long or wide not {how}'

    features = vectoriser.get_feature_names_out()
    if selector is not None:
        features = features[selector.get_support()]
    axis_names = [f'freature_{x + 1}' for x in range(top_n)]

    if len(clf.classes_) > 2:
        results = list()
        for c, coefs in zip(clf.classes_, clf.coef_):
            idx = coefs.argsort()[::-1][:top_n]
            results.extend(tuple(zip([c] * top_n, features[idx], coefs[idx])))
    else:
        coefs = clf.coef_.flatten()
        idx = coefs.argsort()[::-1][:top_n]
        results = tuple(zip([clf.classes_[1]] * top_n, features[idx], coefs[idx]))

    df_lambda = pd.DataFrame(results, columns =  ['sdg', 'feature', 'coef'])

    if how == 'wide':
        df_lambda = pd.DataFrame(
            np.array_split(df_lambda['feature'].values, len(df_lambda) / top_n),
            index = clf.classes_ if len(clf.classes_) > 2 else [clf.classes_[1]],
            columns = axis_names
        )

    return df_lambda

def fix_sdg_name(sdg: str, width: int = 30) -> str:
    sdg_id2name = {
        1: 'GOAL 1: No Poverty',
        2: 'GOAL 2: Zero Hunger',
        3: 'GOAL 3: Good Health and Well-being',
        4: 'GOAL 4: Quality Education',
        5: 'GOAL 5: Gender Equality',
        6: 'GOAL 6: Clean Water and Sanitation',
        7: 'GOAL 7: Affordable and Clean Energy',
        8: 'GOAL 8: Decent Work and Economic Growth',
        9: 'GOAL 9: Industry, Innovation and Infrastructure',
        10: 'GOAL 10: Reduced Inequality',
        11: 'GOAL 11: Sustainable Cities and Communities',
        12: 'GOAL 12: Responsible Consumption and Production',
        13: 'GOAL 13: Climate Action',
        14: 'GOAL 14: Life Below Water',
        15: 'GOAL 15: Life on Land',
        16: 'GOAL 16: Peace and Justice Strong Institutions',
        17: 'GOAL 17: Partnerships to achieve the Goal',
    }
Sdg_id3name = {1 : {goal : “Goal 1:No Poverty”, 

        Target_1_1 : “Target 1.1:eradicate extreme poverty”
        Target_1_2 : “Target 1.2:reduce poverty by atleast 50%”
        Target_1_3 : “Target 1.3:implement social protection systems”
        Target_1_4 : “Target 1.4:equal rights to ownership, basic services, technology and economic resources”
        Target_1_5 : “Target 1.5:build resilience to environmental, economic and social disasters”
        Target_1_6 : “Target 1.6:mobilize resources to implement policies to end poverty”
        Target_1_7 : “Target 1.7:create pro-poor and gender sensitive policy frameworks”
}
}   

Sdg_id4name = {2 : {goal : “Goal 2:Zero Hunger”, 

        Target_2_1 : “Target 2.1:universal access to safe and nutritious food”
        Target_2_2 : “Target 2.2:end all forms of malnutrition”
        Target_2_3 : “Target 2.3:double the productivity and incomes of small-scale food producers”
        Target_2_4 : “Target 2.4:sustainable food production and resilient agricultural practices”
        Target_2_5 : “Target 2.5:maintain the genetic diversity in food production”
        Target_2_6 : “Target 2.6:invest in rural infrastructure, agricultural research, technology and gene banks”
        Target_2_7 : “Target 2.7:prevent agricultural trade restrictions,market distortions and export subsidies”
        Target_2_8 : “Target 2.8:ensure stable food commodity markets and timely access to information”

}
}

Sdg_id5name = {3 : {goal : “Goal 3:Good Health and Well-being”, 

        Target_3_1 : “Target 3.1:reduce maternal mortality”
        Target_3_2 : “Target 3.2:end all preventable deaths under 5 years of age”
        Target_3_3 : “Target 3.3:fight communicable diseases”
        Target_3_4 : “Target 3.4:reduce mortality from non-communicable diseases and promote mental health”
        Target_3_5 : “Target 3.5:prevent and treat substance abuse”
        Target_3_6 : “Target 3.6:reduce road injuries and deaths”
        Target_3_7 : “Target 3.7:universal access to sexual and reproductive care, family planning and education”
        Target_3_8 : “Target 3.8:achieve universal health coverage”
        Target_3_9 : “Target 3.9:reduce illnesses and death from hazardous chemicals and pollution”
        Target_3_A : “Target 3.A:implement the WHO framework convention on tobacco control”
        Target_3_B : “Target 3.B:support research,development and universal access to affordable vaccines and medicines”
        Target_3_C : “Target 3.C:increase health financing and support health workforce in developing countries”
        Target_3_D : “Target 3.D:improve early warning systems for global health risks”
}
}

Sdg_id6name = {4 : {goal : “Goal 4:Quality Education”, 

        Target_4_1 : “Target 4.1:free primary and secondary education”
        Target_4_2 : “Target 4.2:equal access to quality pre-primary education”
        Target_4_3 : “Target 4.3:equal access to affordable technical, vocational and higher education”
        Target_4_4 : “Target 4.4:increase the number of people with relevant skills for financial success”
        Target_4_5 : “Target 4.5:eliminate all discrimination in education”
        Target_4_6 : “Target 4.6:universal literacy and numeracy”
        Target_4_7 : “Target 4.7:education for sustainable development and global citizenship”
        Target_4_8 : “Target 4.8:build and upgrade inclusive and safe schools”
        Target_4_9 : “Target 4.9:expand higher education scholarships for developing countries”
        Target_4_A : “Target 4.A:increase the supply of qualified teachers in developing countries”
}
}
Sdg_id7name = {5 : {goal : “Goal 5:Gender Equality”, 

        Target_5_1 : “Target 5.1:end discrimination against women and girls”
        Target_5_2 : “Target 5.2:end all violence against and exploitation of women and girls”
        Target_5_3 : “Target 5.3:eliminate forced marriages and genital mutilation”
        Target_5_4 : “Target 5.4:value unpaid care and promote shared domestic responsibilites”
        Target_5_5 : “Target 5.5:ensure full participaton in leadership and decision making”
        Target_5_6 : “Target 5.6:universal access to reproductive health and rights”
        Target_5_7 : “Target 5.7:equal rights to economic resources, property ownership and financial services”
        Target_5_8 : “Target 5.8:promote empowerment of women through technology”
        Target_5_9 : “Target 5.9:adopt and strengthen policies and enforceable legistlation for gender equality”
}
}
Sdg_id8name = {6 : {goal : “Goal 6:Clean Water and Sanitation”, 

        Target_6_1 : “Target 6.1:safe and affordable drinking water”
        Target_6_2 : “Target 6.2:end open defecation and provide access to sanitation and hygiene”
        Target_6_3 : “Target 6.3:improve water quality, wastewater treatment and safe reuse”
        Target_6_4 : “Target 6.4:increase water-use efficiency and ensure freshwater supplies”
        Target_6_5 : “Target 6.5:implement integrated water resource management”
        Target_6_6 : “Target 6.6:protect and restore water related eco-systems”
        Target_6_7 : “Target 6.7:expand water and sanitation support to developing countries”
        Target_6_8 : “Target 6.8:support local engagement in water and sanitation management”
}
} 
Sdg_id9name = {7 : {goal : “Goal 7:Affordable and Clean Energy”, 

        Target_7_1 : “Target 7.1:universal access to mordern energy”
        Target_7_2 : “Target 7.2:increase global percentage of renewable energy”
        Target_7_3 : “Target 7.3:double the improvement in energy efficiency”
        Target_7_4 : “Target 7.4:promote access to research, technology and investments in clean energy”
        Target_7_5 : “Target 7.4:expand and upgrade energy services for developing countries”

}
}   
Sdg_id10name = {8 : {goal : “Goal 8:Decent Work and Economic Growth”, 

        Target_8_1 : “Target 8.1:sustainable economic growth”
        Target_8_2 : “Target 8.2:diversify, innovate and upgrade for economic productivity”
        Target_8_3 : “Target 8.3:promote policies to support job creation and growing enterprises”
        Target_8_4 : “Target 8.4:improve resource efficiency in consumption and production”
        Target_8_5 : “Target 8.5:full employment and decent work with equal pay”
        Target_8_6 : “Target 8.6:promote youth employment education and training”
        Target_8_7 : “Target 8.7:end mordern slavery,trafficking and child labour”
        Target_8_8 : “Target 8.8:protect labour rights and promote safe working environments”
        Target_8_9 : “Target 8.9:promote beneficial and sustainable tourism”
        Target_8_A : “Target 8.A:universal access to banking, insurance and financial services”
        Target_8_B : “Target 8.B:increase aid for trade support”
        Target_8_C : “Target 8.C:develop a global youth employment strategy”
}
}
Sdg_id11name = {9 : {goal : “Goal 9:Industry, Innovation and Infrastructure”, 

        Target_9_1 : “Target 9.1:develop sustainable resilient and inclusive infrastructures”
        Target_9_2 : “Target 9.2:promote inclusive and sustainable industrialization”
        Target_9_3 : “Target 9.3:increase access to financial services and markets”
        Target_9_4 : “Target 9.4:upgrade all industries and infrastructures for sustainability”
        Target_9_5 : “Target 9.5:enhance research and upgrade industrial technologies”
        Target_9_6 : “Target 9.6:facilitate sustainable infrastructure development for developing countries”
        Target_9_7 : “Target 9.7:support domestic technology development and industrial diversification”
        Target_9_8 : “Target 9.8:universal access to information and communications technology”
}
}
Sdg_id12name = {10 : {goal : “Goal 10:Reduced Inequality”, 

        Target_10_1 : “Target 10.1:reduce income inequalities”
        Target_10_2 : “Target 10.2:promote universal social, economic and political inclusion”
        Target_10_3 : “Target 10.3:ensure equal opportunities and end discrimination”
        Target_10_4 : “Target 10.4:adopt fiscal and social policies that promote equality”
        Target_10_5 : “Target 10.5:improved regulation of global financial markets and instituitions”
        Target_10_6 : “Target 10.6:enhanced representation for developing countries in financial instituitions”
        Target_10_7 : “Target 10.7:responsible and well managed migration policies”
        Target_10_8 : “Target 10.8:special and differential treatment for developing countries”
        Target_10_9 : “Target 10.9:encourage development assistance and investment in least developed countries”
        Target_10_A : “Target 10.A:reduce transactions costs for migrant remittances”

}
}
Sdg_id13name = {11 : {goal : “Goal 11:Sustainable Cities and Communities”, 

        Target_11_1 : “Target 11.1:safe and affordable housing”
        Target_11_2 : “Target 11.2:affordable and sustainable transport systems”
        Target_11_3 : “Target 11.3:inclusive and sustainable urbanization”
        Target_11_4 : “Target 11.4:protect the worlds cultural and natural heritage”
        Target_11_5 : “Target 11.5:reduce the adverse effects on natural disasters”
        Target_11_6 : “Target 11.6:reduce the environmental impact of cities”
        Target_11_7 : “Target 11.7:provide access to safe and inclusive green and public spaces”
        Target_11_8 : “Target 11.8:strong national and regional development planning”
        Target_11_9 : “Target 11.9:implement policies for inclusion,resource efficiency and disaster risk reduction”
        Target_11_A : “Target 11.A:support least developed countries in sustainable and resilient building”



}
}
Sdg_id14name = {12 : {goal : “Goal 12:Responsible Consumption and Production”, 

        Target_12_1 : “Target 12.1:implement the 10-year sustainable consumption and production framework”
        Target_12_2 : “Target 12.2:sustainable management and use of natural resources”
        Target_12_3 : “Target 12.3:have global per capita food waste”
        Target_12_4 : “Target 12.4:responsible management of chemicals and waste”
        Target_12_5 : “Target 12.5:substantially reduce waste generation”
        Target_12_6 : “Target 12.6:encourage companies to adopt sustainable practices and sustainability reporting”
        Target_12_7 : “Target 12.7:promote sustainable public procurement practices”
        Target_12_8 : “Target 12.8:promote universal understanding of sustainable lifestyles”
        Target_12_9 : “Target 12.9: support developing countries scientific and technological capacity for sustainable consumption and production”
        Target_12_A : “Target 12.A: develop and implement tools to monitor sustainable tourism”
        Target_12_B : “Target 12.B: remove market distortions that encourage wasteful consumption”

}
}
Sdg_id15name = {13 : {goal : “Goal 13:Climate Action”, 

        Target_13_1 : “Target 13.1:strengthen resilience and adaptive capacity to climate related disasters”
        Target_13_2 : “Target 13.2:integrate climate change measures into policies and planning”
        Target_13_3 : “Target 13.3:build knowledge and capacity to meet climate change”
        Target_13_4 : “Target 13.4:implement the un framework convention on climate change”
        Target_13_5 : “Target 13.5:promote mechanisms to raise capacity for planning and management”
        
}
}

Sdg_id16name = {14 : {goal : “Goal 14:Life Below Water”, 

        Target_14_1 : “Target 14.1:reduce marine pollution”
        Target_14_2 : “Target 14.2:protect and restore ecosystems”
        Target_14_3 : “Target 14.3:reduce ocean acidification”
        Target_14_4 : “Target 14.4:sustainable fishing”
        Target_14_5 : “Target 14.5:conserve coastal and marine areas”
        Target_14_6 : “Target 14.6:end subsidies contributing to overfishing”
        Target_14_7 : “Target 14.7:increase the economic benefits from sustainable use of marine resources”
        Target_14_8 : “Target 14.8:increase scientific knowledge, research and technology for ocean health”
        Target_14_9 : “Target 14.9:support small scale fishers”
        Target_14_A : “Target 14.9:implement and enforce international sea law”

        

}
}
Sdg_id17name = {15 : {goal : “Goal 15:Life on Land”, 

        Target_15_1 : “Target 15.1:conserve and restore terrestrial and freshwater ecosystems”
        Target_15_2 : “Target 15.2:end deforestation and restore degraded forests”
        Target_15_3 : “Target 15.3:end desertification and restore degraded land”
        Target_15_4 : “Target 15.4:ensure conservation of mountain ecosystems”
        Target_15_5 : “Target 15.5:protect biodiversity and natural habitats”
        Target_15_6 : “Target 15.6:promote access to genetic resources and fair sharing of the benefits”
        Target_15_7 : “Target 15.7:eliminate poaching and trafficking of protected species”
        Target_15_8 : “Target 15.8:prevent invasive alien species on land and in water ecosystems”
        Target_15_9 : “Target 15.9:integrate ecosystem and biodiverstiy in government planning”
        Target_15_A : “Target 15.A:increase financial resources to conserve and sustainably use ecosystem and biodiversity”
        Target_15_B : “Target 15.B:finance and incentivize sustainable forest management”
        Target_15_B : “Target 15.C:combat global poaching and trafficking”




}
}
Sdg_id18name = {16 : {goal : “Goal 16:Peace and Justice Strong Institutions”, 

        Target_16_1 : “Target 16.1:reduce violence everywhere”
        Target_16_2 : “Target 16.2:protect children from abuse, exploitation,trafficking and violence”
        Target_16_3 : “Target 16.3:promote the rule of law and ensure equal access to justice”
        Target_16_4 : “Target 16.4:combat organized crime and illicit financial and arms flows”
        Target_16_5 : “Target 16.5:substaintially reduce corruption and bribery”
        Target_16_6 : “Target 16.6:develop effective accountable and transparent instituitions”
        Target_16_7 : “Target 16.7:ensure responsive, inclusive and representative decision making”
        Target_16_8 : “Target 16.8:strengthen the participation in global governance”
        Target_16_9 : “Target 16.9:provide universal legal identity”
        Target_16_A : “Target 16.A:enusure public access to information and protect fundamental freedoms”
        Target_16_B : “Target 16.B:strengthen national instituitions to prevent violence and combat terrorism and crime”
        Target_16_C : “Target 16.C:promote and enforce non-discriminatory laws and policies”
}
}
Sdg_id19name = {17 : {goal : “Goal 17:Partnerships to achieve the Goal”, 

        Target_17_1 : “Target 17.1:mobilize resources to improve domestic revenue collection”
        Target_17_2 : “Target 17.2:implement all development assistance commitments”
        Target_17_3 : “Target 17.3:mobilize financial resources for developing countries”
        Target_17_4 : “Target 17.4:assist developing countries in attaining debt sustainability”
        Target_17_5 : “Target 17.5:invest in least developed countries”
        Target_17_6 : “Target 17.6:knowledge sharing and cooperation for access to science, technology and innovation”
        Target_17_7 : “Target 17.7:promote sustainable technologies to developing countries”
        Target_17_8 : “Target 17.8:strengthen the science, technology and innovation capacity for least developed countries”
        Target_17_9 : “Target 17.9:enhance sdg capacity in developing countries”
        Target_17_A : “Target 17.A:promote a universal trading system under the WTO”
        Target_17_B : “Target 17.B:increase the exports of developing countries”
        Target_17_C : “Target 17.C:remove trade barriers for least developed countries”
        Target_17_D : “Target 17.D:enhance global macroeconomic stability”
        Target_17_E : “Target 17.E:enhance policy coherence for sustainability development”
        Target_17_F : “Target 17.E:respect national leadership to implement policies for the sustainable development goals”
        Target_17_G : “Target 17.G:enhance the global patnership for sustainable development”
        Target_17_H : “Target 17.H:encourage effective partnerships”
        Target_17_I : “Target 17.I:enhance availability of reliable data”
        Target_17_J : “Target 17.J:further develop measurements of progress”


}
}
    
     
     















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































    name = sdg_id2name[int(sdg)]
    return '<br>'.join(wrap(name, 30))
