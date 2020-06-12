from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import statistics
import scipy 
from scipy import optimize  
from condensation_temperature import * 
from tqdm import tqdm

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 12
plt.rcParams["figure.figsize"] = fig_size

t= Table.read('solar_twins_data.fits') #fits file as table 

exclusions = ['HIP19911', 'HIP108158', 'HIP109821', 'HIP115577', 'HIP14501', 'HIP28066', 'HIP30476',
              'HIP33094', 'HIP65708', 'HIP73241', 'HIP74432', 'HIP64150']
for u in exclusions:
    for i, txt in enumerate(t['star_name']):
        if txt == u:
            t.remove_row(i)

for i, words in enumerate(t['Fe']):
   t['Fe'][i] = 0

t['O'][8] = 0
t['O_err'][8] = 10**6

def star_table(star):
    tableco = t.copy()
    tableco.remove_column('O')
    tableco.remove_column('C')
    tableco.remove_column('O_err')
    tableco.remove_column('C_err')

    tcvals= tc_map.copy()
    del tcvals['O']
    del tcvals['C']

    for i, txt in enumerate(tableco['star_name']):
        if txt == star:
            tbl = tableco[i] #inputted star's row)

    star_elements =[]
    elnames = tbl.columns[3:64]
    for n in elnames:
        if len(n) < 3 :
            star_elements.append(n)
            star_elements #list of elements in that star
    
    star_abundance = []
    for n in star_elements:
        star_abundance.append(tbl[n])
        star_abundance #list of element abundances
        
    star_con_temp = []
    for n in star_elements:
        star_con_temp.append(tc_map[n])
        star_con_temp #condensation temperatures for stellar elements
    
    star_error_elements = []
    for r in elnames:
        if len(r) > 3 :
            star_error_elements.append(r) #list of elements recorded in star

    el_error = []
    for k in star_error_elements:
        el_error.append(tbl[k])
        el_error #list of error values for elements

    atmnum =[26,11,12,13,14,16,20,23,25,27,28,29,30,38,39,40,56,57,58,59,60,62,63,64,66,22,21,24]

    for x, txt in enumerate(star_abundance):
        if (math.isnan(txt) == True):
            del star_elements[x]
            del star_abundance[x]
            del star_con_temp[x]
            del el_error[x]
            del atmnum[x]

    star_table = Table([star_elements, star_abundance, el_error, star_con_temp, atmnum], names=('Element', 'Abundance', 'Abundance Error','Condensation Temp', 'Atomic Number')) #table of temperature vs abundance for elements 
    return star_table

#function for returning the best slope and intercept using linear algebra : Hogg eq 5 
#[m b] = [A^T C^-1 A]^-1 [A^T C^-1 Y]
def find_m_b(x,y,err): 
    #C 
    errorsq = np.square(err)
    C = np.diag(errorsq)
    
    #A
    xb = ([1] * len(x))
    mata = []   
    for z, txt in enumerate(x):
        mata.append(x[z])
        mata.append(xb[z])
    A= np.matrix(mata).reshape((len(x), 2))
    
    #plugging in 
    At = np.transpose(A)
    invC = np.linalg.inv(C)
    pt1 = np.dot(At, np.dot(invC,A))
    invpt1= np.linalg.inv(pt1)
    pt2 = np.dot(At, np.dot(invC, y)).T
    cov = np.dot(invpt1, pt2)
        
    m_= float(cov[0])
    b_= float(cov[1])
    return m_,b_ 

#jackknife method for determining other possible values of m and b 
def jackknifemb(_tp,_ab,_er):
    N=1000
    l=list(np.copy(_tp))
    k=list(np.copy(_ab))
    s=list(np.copy(_er))
    jackm= []
    jackb= [] 
    h=0
    
    #leaving out one point from data set and calculating m, b for each instance
    while h<N:
        w = random.randint(0, (len(_tp)-1))
        del l[w]
        del k[w]
        del s[w] #removing one data set from lists 
    
        jk_mb = find_m_b(l,k,s)
        jk_m = jk_mb[0]
        jk_b = jk_mb[1]

        jackm.append(jk_m) #alternate m values
        jackb.append(jk_b) #alternate b values
            
        l=list(np.copy(_tp)) #adding value back in for next round 
        k=list(np.copy(_ab)) 
        s=list(np.copy(_er))
        h=h+1 
        
    return jackm, jackb

def stellar_abundance_plot(star): 
    table = star_table(star)
    temp= np.array(table.columns[3])
    abund = np.array(table.columns[1])
    error = np.array(table.columns[2])
    elements = np.array(table.columns[0])
    
    plt.ioff()
    fig, ax = plt.subplots()

    ax.scatter(temp, abund)
    ax.set_xlabel('Tc',fontsize='xx-large', family='sans-serif')
    ax.set_ylabel('Element Abundance', fontsize='xx-large', family='sans-serif')
    ax.set_title('Temperature vs Element Abundance for {0}'.format(star), fontsize= 'xx-large', family='sans-serif')

    #point labels    
    for i, txt in enumerate(elements):
            ax.annotate(txt, xy=(temp[i], abund[i]), xytext=(-13,-6), 
                textcoords='offset points', ha='center', va='bottom')
    
    #alternate best fit lines
    jk= jackknifemb(temp, abund, error)
    for i, txt in enumerate (jk[0]):
        plot_xs = np.arange(0, 1750, .1)
        ax.plot(plot_xs, jk[0][i] * plot_xs + (jk[1][i]), color = 'lightgray', linewidth=0.1)

    #error bars
    ax.errorbar(temp, abund, yerr= error, fmt='o', color='black',
                 ecolor='lightsteelblue', elinewidth=3,capsize=0)
    
    #line of best fit m,b values
    mb = find_m_b(temp, abund, error) 
    plot_xs = np.arange(0, 1750, .1)
    ax.plot(plot_xs, (mb[0]) * plot_xs + (mb[1]), color='teal')

    fig.savefig(star+'.png')
    plt.close(fig)

def abund_plot_noCO(star): 
    table = star_table(star)
    temp= np.array(table.columns[3])
    abund = np.array(table.columns[1])
    error = np.array(table.columns[2])
    elements = np.array(table.columns[0])
    
    C_O_removed_error = [] #lists without C or O data -- outliers
    C_O_removed_temp = []
    C_O_removed_abund = []
    for h, name in enumerate(elements):
        if name != 'C':
            if name != 'O':
                C_O_removed_error.append(error[h])
                C_O_removed_temp.append(temp[h])
                C_O_removed_abund.append(abund[h])
    
    plt.ioff()
    fig, ax = plt.subplots()

    #point labels
    for i, txt in enumerate(elements):
            ax.annotate(txt, xy=(temp[i], abund[i]), xytext=(-13,-6),
                textcoords='offset points', ha='center', va='bottom')
    
    #alternate best fit lines  
    jk= jackknifemb(C_O_removed_temp, C_O_removed_abund, C_O_removed_error)
    for i, txt in enumerate (jk[0]):
        plot_xs = np.arange(0, 1750, .1)
        ax.plot(plot_xs, jk[0][i] * plot_xs + (jk[1][i]), color = 'lightgray', linewidth=0.1)
    
    #error bars 
    for u, name in enumerate(elements): #plotting points, with C and O in different colors
        if name == 'C':
            ax.errorbar(temp[u], abund[u], yerr= error[u], fmt='o', color='blue',
                 ecolor='lightsteelblue', elinewidth=3, capsize=0)
        elif name == 'O' :
            ax.errorbar(temp[u], abund[u], yerr= error[u], fmt='o', color='blue',
                 ecolor='lightsteelblue', elinewidth=3, capsize=0)
        else:
            ax.errorbar(C_O_removed_temp, C_O_removed_abund, yerr= C_O_removed_error, fmt='o', color='black',
                 ecolor='lightsteelblue', elinewidth=3, capsize=0)
    
    #plot labels
    ax.set_xlabel('Tc (K)',fontsize= 30, family='sans-serif')
    ax.set_ylabel('Element Abundance (dex)', fontsize=30, family='sans-serif')
    ax.set_title('Temperature vs Element Abundance for {0}'.format(star), fontsize=30, family='sans-serif')
    plt.xticks(size = 15)
    plt.yticks(size = 15)

    #line of best fit m,b values
    mb = find_m_b(C_O_removed_temp, C_O_removed_abund, C_O_removed_error)    
    plot_xs = np.arange(0, 1750, .1)
    ax.plot(plot_xs, (mb[0]) * plot_xs + (mb[1]), color='teal') 
    
    fig.savefig(star+'noco.png')
    plt.close(fig)

#chi squared, Hogg 2010 eq 7 :  [Y - AX]^T C^-1 [Y - AX]                                                                                                                                                   
def chisquared(param, x, y, erro):
    for h, txt in enumerate(y): #removing nan values                                                                                                                                                      
        if (math.isnan(txt) == True):
            del x[h]
            del y[h]
            del erro[h]

    (m,b) = param #for optimization - m and b    
    X = (m,b)

    #A                                                                                                                                                                                                   
    ab = ([1] * len(x))
    Amat = []
    for z, txt in enumerate(x):
        Amat.append(x[z])
        Amat.append(ab[z])
    A= np.array(Amat).reshape((len(x), 2))

    #C
    errorsq = np.square(erro)
    C = np.diag(errorsq)
    invsC = np.linalg.inv(C)

    #plugging in                                                                                                                                                                                          
    AT= np.transpose(A)
    AX = np.dot(A,X)
    yax = (y - AX)
    yaxT = np.transpose(yax)
    yaxTinvsC = np.dot(yaxT, invsC)

    chisq = (np.dot(yaxTinvsC, yax))
    return (chisq)

def covmatrix(x, y, error): #Hogg 2010 eq 18     
    #removing nan data 
    for h, txt in enumerate(y):
        if (math.isnan(txt) == True):
            del x[h]
            del y[h]
            del error[h]
    
    #C    
    errororsq = np.square(error)
    errororC = np.diag(errororsq)
    abu = ([1] * len(x))
    
    #A 
    axer = np.copy(x)
    matri = []   
    for z, txt in enumerate(axer):
        matri.append(axer[z])
        matri.append(abu[z])        
    aa= np.matrix(matri).reshape((len(x), 2))
    
    #transpose of A and inverse of C, then plugged in 
    Att = np.transpose(aa)
    inverrororC = np.linalg.inv(errororC)
    inbrackets = np.dot(Att, np.dot(inverrororC, aa))
    
    covmatrix = np.linalg.inv(inbrackets)
    #covmatrix = [σ^2m, σmb, σmb, σ^2b]
    return covmatrix

def standardslopeerror(x, y, err):
    for h, txt in enumerate(y):
        if (math.isnan(txt) == True):
            del x[h]
            del y[h]
            del err[h]
    #C       
    errorsq = np.square(err)
    errorC = np.diag(errorsq)
    
    #A
    abu = ([1] * len(x))
    atemper = np.copy(x)
    matri = []
    for z, txt in enumerate(atemper):
        matri.append(atemper[z])
        matri.append(abu[z])
    aa= np.matrix(matri).reshape((len(x), 2))
    
    #plugging in 
    Att = np.transpose(aa)
    inverrorC = np.linalg.inv(errorC)
    prt1 = np.dot(Att, np.dot(inverrorC,aa))
    invt1= np.linalg.inv(prt1)
    prt2 = np.dot(Att, np.dot(inverrorC, y)).T
    covar = np.dot(invt1, prt2)
        
    _m_= float(covar[0])
    _b_= float(covar[1]) #standard slope, intercept values found with linalg 
    
    inbrackets = np.dot(Att, np.dot(inverrorC, aa))
    sserror = np.linalg.inv(inbrackets)
    #sserror = [σ^2m, σmb, σmb, σ^2b]
    sse = np.sqrt(sserror[0,0]) #standard slope error
    return sse

def standardintercepterror(x,y, err):
    for h, txt in enumerate(y):
        if (math.isnan(txt) == True):
            del x[h]
            del y[h]
            del err[h]
    #C       
    errorsq = np.square(err)
    errorC = np.diag(errorsq)
    
    abu = ([1] * len(x))
    atemper = np.copy(x)
    matri = []
    for z, txt in enumerate(atemper):
        matri.append(atemper[z])
        matri.append(abu[z])
    aa= np.matrix(matri).reshape((len(x), 2))
    
    Att = np.transpose(aa)
    inverrorC = np.linalg.inv(errorC)
    prt1 = np.dot(Att, np.dot(inverrorC,aa))
    invt1= np.linalg.inv(prt1)
    prt2 = np.dot(Att, np.dot(inverrorC, y)).T
    covar = np.dot(invt1, prt2)
        
    _m_= float(covar[0])
    _b_= float(covar[1]) #standard slope, intercept values found with linalg 
    
    inbrackets = np.dot(Att, np.dot(inverrorC, aa))
    sserror = np.linalg.inv(inbrackets)
    #sserror = [σ^2m, σmb, σmb, σ^2b]
    sie = np.sqrt(sserror[1,1]) #standard slope error
    return sie

def error_table(tp, ab, er):
    jackm = jackknifemb(tp, ab, er)[0]
    jackb = jackknifemb(tp, ab, er)[1]
    
    slopeer = standardslopeerror(tp,ab,er)
    interer= standardintercepterror(tp,ab,er)
    slopeintercept = find_m_b(tp,ab,er)
    
    error_type = ['slope', 'intercept']
    a = [slopeintercept[0], slopeintercept[1]]
    c = [statistics.stdev(jackm),statistics.stdev(jackb)]
    d = [slopeer, interer]
    tab = Table([error_type,a, c, d], names=('error type', 'value','standard dev', 
                                              'linear algebra uncertainty'))
    return tab

def residuals(x, y, error):
    mborig = find_m_b(x, y, error)
    m = mborig[0]
    b = mborig[1]

    predicted_values = [] #y values from slope                            
    pv = 0
    for u in x:
        pv = (m*u) + b
        predicted_values.append(pv)
        pv = 0

    prev = np.array(predicted_values)
    abu = np.array(y)
    diff = abu - prev #difference between slope and measured values  
    return diff

def abudiff(star): #plots for abundance differences     
    table = star_table(star)
    temp= np.array(table.columns[3])
    error = np.array(table.columns[2])
    abund = np.array(table.columns[1])
    elements = np.array(table.columns[0])
    
    diff = residuals(temp, abund, error)
    plt.ioff()
    fig, ax = plt.subplots()

    ax.scatter(temp, diff)
    ax.set_xlabel('Tc (K)',fontsize='xx-large', family='sans-serif')
    ax.set_ylabel('Tc-Corrected Abundance (dex)', fontsize='xx-large', family='sans-serif')
    ax.set_title('Temperature vs Abundance for {0}'.format(star), fontsize= 'xx-large', family='sans-serif')

    #point labels
    for i, txt in enumerate(elements):
            ax.annotate(txt, xy=(temp[i], diff[i]), xytext=(-13,-6),
                textcoords='offset points', ha='center', va='bottom')

    jk= jackknifemb(temp, diff, error)
    for i, txt in enumerate (jk[0]):
        plot_xs = np.arange(0, 1750, .1)
        ax.plot(plot_xs, jk[0][i] * plot_xs + (jk[1][i]), color = 'lightgray', linewidth=0.1)

    #error bars
    ax.errorbar(temp, diff, yerr= error, fmt='o', color='black',
                 ecolor='lightsteelblue', elinewidth=3,capsize=0)

    #line of best fit m,b values
    mb = find_m_b(temp, diff, error)
    plot_xs = np.arange(0, 1750, .1)
    ax.plot(plot_xs, (mb[0]) * plot_xs + (mb[1]), color='teal')

    fig.savefig('tcremoved'+ star + '.png')
    plt.close(fig)

def abudiff_noCO(star): #plots for abundance differences   
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    table = star_table(star)
    temp= np.array(table.columns[3])
    abund = np.array(table.columns[1])
    error = np.array(table.columns[2])
    elements = np.array(table.columns[0])
    
    diff = residuals(temp, abund, error)
    
    C_O_removed_error = [] #lists without C or O data -- outliers                                                                                                                                           
    C_O_removed_temp = []
    C_O_removed_abund = []
    C_O_removed_diff = []
    for h, name in enumerate(elements):
        if name != 'C':
            if name != 'O':
                C_O_removed_error.append(error[h])
                C_O_removed_temp.append(temp[h])
                C_O_removed_abund.append(abund[h])
                C_O_removed_diff.append(diff[h])
                
    plt.ioff()
    fig, ax = plt.subplots()

     #point labels
    for n, txt in enumerate(elements):
        if txt == 'C':
            ax.annotate(txt, xy=(temp[n], abund[n]), xytext=(-13,-6),
                            textcoords='offset points', ha='center', va='bottom')
        elif txt == 'O' :
            ax.annotate(txt, xy=(temp[n], abund[n]), xytext=(-13,-6),
                            textcoords='offset points', ha='center', va='bottom')
        else:
            ax.annotate(txt, xy=(temp[n], diff[n]), xytext=(-13,-6),
                            textcoords='offset points', ha='center', va='bottom')

    #alternate best fit lines                                                                                                                                                                               
    jk= jackknifemb(C_O_removed_temp, C_O_removed_diff, C_O_removed_error)
    for i, txt in enumerate (jk[0]):
        plot_xs = np.arange(0, 1750, .1)
        ax.plot(plot_xs, jk[0][i] * plot_xs + (jk[1][i]), color = 'lightgray', linewidth=0.1)
        
    #error bars                                                                                                                                                                                             
    for u, name in enumerate(elements): #plotting points, with C and O in different colors                                                                                                                  
        if name == 'C':
            ax.errorbar(temp[u], abund[u], yerr= error[u], fmt='o', color='blue',
                 ecolor='lightsteelblue', elinewidth=3, capsize=0)
        elif name == 'O' :
            ax.errorbar(temp[u], abund[u], yerr= error[u], fmt='o', color='blue',
                 ecolor='lightsteelblue', elinewidth=3, capsize=0)
        else:
            ax.errorbar(C_O_removed_temp, C_O_removed_diff, yerr= C_O_removed_error, fmt='o', color='black',
                 ecolor='lightsteelblue', elinewidth=3, capsize=0)

    #plot labels                                                                                                                                                                                            
    ax.set_xlabel('Tc (K)',fontsize= 30, family='sans-serif')
    ax.set_ylabel('Element Abundance (dex)', fontsize=30, family='sans-serif')
    ax.set_title('Temperature vs Element Abundance for {0}'.format(star), fontsize= 30, family='sans-serif')

    
    #line of best fit m,b values                                                                                                                                                                            
    mb = find_m_b(C_O_removed_temp, C_O_removed_diff, C_O_removed_error)
    plot_xs = np.arange(0, 1750, .1)
    ax.plot(plot_xs, (mb[0]) * plot_xs + (mb[1]), color='teal')

    fig.savefig(star+'noco_diff.png')
    plt.close(fig)


def lnL(param, x, y, error): # Hogg: lnL = -1/2 sum [(y- (mx+b))^2/(err^2 + d^2)] -1/2 sum ln(err^2 + d^2)
    x = np.array(x)
    y = np.array(y)
    error = np.array(error)
    (m,b,d) = param 

    sum1 = 0
    for j, val in enumerate(x):
        mxb = m*val + b
        value = ((y[j] - mxb)**2)/((error[j]**2) + (d**2))
        sum1 = sum1 + value 
    sum1 = sum1 *(-1/2)
    
    num = 0
    for e, ooh in enumerate(error):
        val2 = np.log((ooh**2) + (d**2))
        num = num + val2 
    num = num * (-1/2)
    
    lnL = sum1 + num
    return lnL

def nlnL(param, x, y, error):
    nlnL= lnL(param, x,y, error)
    return - nlnL

#2D uncertainties: lnL = K -  sum [delta^2 / 2* sum^2]  
def twodnlnL(param, x, y, errx, erry):
    K = 0
    delt = delta(param, x, y)
    (m,b,d) = param

    #Σ^2 = vˆT Si v  
    theta = np.arctan(m)
    v = np.array([- np.sin(theta), np.cos(theta)]) # v = [− sin θ,  cos θ]
    
    var = np.zeros_like(erry)
    for i, (dy, dx) in enumerate(zip(erry, errx)):
        cov = np.eye(2)
        cov[0,0] = dx**2
        cov[1,1] = dy**2 
        var[i] = np.dot(v.T, np.dot(cov, v))
        
    sigmasq = var + np.exp(2.*d)
    return 0.5 * np.sum(delt**2/sigmasq + np.log(sigmasq))

def delta(param, x, y): # Hogg 2010 eqn 30 : ∆i = (vˆT Zi) − b cos θ   
    (m,b,d) = param
    theta = np.arctan(m)
    v = np.array([- np.sin(theta), np.cos(theta)]) # v = [− sin θ  cos θ]   

    disp = np.zeros_like(y)
    for i, (ys, xs) in enumerate(zip(y, x)):
        z0 = np.asarray([0.0, b])
        zi = np.asarray([xs, ys])
        disp[i] = np.dot( v, zi - z0 )
    return disp

def dm(param, x, y, error):
    x = np.array(x)
    y = np.array(y)
    error = np.array(error)
    (m,b,d) = param 

    sum1 = 0
    for c, valu in enumerate(x): # sum (1/(d^2 + err^2) * 2(y-(mx+b)) * -x
        mxb = m*valu + b
        value = (1/((error[c]**2) + (d**2)) * 2 *(y[c] - mxb) * -valu)
        sum1 = sum1 + value 
    sum1 = sum1 * (1/2)
    
    sum2 = 0
    for e, ooh in enumerate(error): # 1/2 * sum (ln(err^2 + d^2))
        val2 = np.log((ooh**2) + (d**2))
        sum2 = sum2 + val2 
    sum2 = sum2 * (1/2)
    
    return sum1 - sum2

def db(param, x, y, error):
    x = np.array(x)
    y = np.array(y)
    error = np.array(error)
    (m,b,d) = param 

    sum1 = 0
    for c, valu in enumerate(x): # sum (1/(d^2 + err^2) * 2(y-(mx+b))
        mxb = m*valu + b
        value = (1/((error[c]**2) + (d**2)) * 2 *(y[c] - mxb))
        sum1 = sum1 + value 
    sum1 = sum1 * (1/2)
    
    sum2 = 0
    for e, ooh in enumerate(error): # 1/2 * sum (ln(err^2 + d^2))
        val2 = np.log((ooh**2) + (d**2))
        sum2 = sum2 + val2 
    sum2 = sum2 * (1/2)
    
    return sum1 - sum2

def dd(param, x, y, error):
    x = np.array(x)
    y = np.array(y)
    error = np.array(error)
    (m,b,d) = param 

    sum1 = 0
    for c, valu in enumerate(x): # sum (2d/(d^2 + err^2)^2 * (y-(mx+b))^2 
        mxb = m*valu + b
        value = (((2*d)/(((error[c]**2) + (d**2))**2)) * (y[c] - mxb)**2)
        sum1 = sum1 + value 
    sum1 = sum1 * (1/2)
    
    sum2 = 0
    for e, ooh in enumerate(error): # 1/2 * sum (1/(err^2 + d^2)  * 2d)
        val2 = (2*d)/((ooh**2) + (d**2))
        sum2 = sum2 + val2 
    sum2 = sum2 * (1/2)
    
    return sum1 - sum2

def delta_minimized(table, element):
    x0 = (1, 0, -1)
    deltatemp = []
    deltanotemp = []
    
    ages = table['age']
    age_error = table['age_err']
    abundance_temp = table[element]
    abundance_error = table[element + '_err']
    abundance_notemp = residuals(ages, abundance_temp, abundance_error)
    
    #BEFORE REMOVING TEMP TRENDS
    delt_temp = scipy.optimize.minimize(twodnlnL, x0, args = (ages, abundance_temp, age_error, abundance_error))
    tvalue = delt_temp['x'][2]
    deltatemp.append(tvalue)

    #AFTER REMOVING TEMP
    delt_notemp = scipy.optimize.minimize(twodnlnL, x0, args = (ages, abundance_notemp, age_error, abundance_error))
    value = delt_notemp['x'][2]
    deltanotemp.append(value)
        
    return deltatemp , deltanotemp

def jackknife_delta(x,y,erx, ery):
    N=100
    l=list(np.copy(x))
    k=list(np.copy(y))
    s=list(np.copy(erx))
    t=list(np.copy(ery))
    jacktemp= []
    jacknotemp= [] 
    h=0
    
    #leaving out one point from data set and calculating delta for each instance
    while h<N:
        w = random.randint(0, (len(x)-1))
        del l[w]
        del k[w]
        del s[w] 
        del t[w] #removing one data set from lists 
    
        #BEFORE REMOVING TEMP TRENDS
        x0 = (1, 0, -1)
        delt_temp = scipy.optimize.minimize(twodnlnL, x0, args = (l,k,s,t))
        tvalue = delt_temp['x'][2]
        jacktemp.append(tvalue)
        
        #AFTER REMOVING TEMP
        abundance_notemp = residuals(l, k, t)
        delt_notemp = scipy.optimize.minimize(twodnlnL, x0, args = (l, abundance_notemp, s, t))
        value = delt_notemp['x'][2]
        jacknotemp.append(value)
            
        l=list(np.copy(x)) #adding value back in for next round 
        k=list(np.copy(y)) 
        s=list(np.copy(erx))
        t=list(np.copy(ery))
        h=h+1 
        
    return jacktemp, jacknotemp

def age_abund_plot(table):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    star_elements = []
    for n in t.colnames:
        if len(n) < 3 :
            star_elements.append(n) #list of elements in table
    elements = np.array(star_elements)
    deltemp=[]
    slopes = []
    for y, ele in tqdm(enumerate(elements)):
        deltemp.append(delta_minimized(t, ele)[0])
    
        x0 = (.01, .03, .07) #initial guess
        restemp = scipy.optimize.minimize(twodnlnL, x0, args = (t['age'], t[ele], t['age_err'], t[ele + '_err']))
    
        plt.ioff()
        fig, ax = plt.subplots()
    
        ax.scatter(t['age'], t[ele], c='rebeccapurple') 
        ax.set_xlabel('Age',fontsize='xx-large')
        ax.set_ylabel(ele + '/Fe', fontsize='xx-large')
        ax.set_title(ele + ' Abundance vs. Stellar Age', fontsize='xx-large')

        #line of best fit
        mb = find_m_b(t['age'], t[ele], t[ele + '_err'])
        slopes.append(mb[0])

        for i, txt in enumerate (t[ele]):
            plot_xs = np.arange(0, 9, .01)
            ax.plot(plot_xs, mb[0] * plot_xs + (mb[1]), color = 'olivedrab', linewidth=1)
    
       #point labels
        #for i, txt in enumerate(t['star_name']): 
                #ax.annotate(txt, xy=(t['age'][i], t[ele][i]), xytext = (-5,5), fontsize='small', 
                    #textcoords='offset points', ha='center', va='bottom')
    
        mbtemp= restemp['x']
        plot_xs = np.arange(0, 9, .1)
        ax.plot(plot_xs, mbtemp[0] * plot_xs + (mbtemp[1]), color = 'black', linewidth=1)
        #ax.text(2,.15, deltemp[y] ,horizontalalignment='right', fontsize=12)
    
        fig.savefig(ele +'_age.png')
        plt.close(fig) 
    #return slopes

def age_abund_plot_no_temp(table):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    star_elements = []
    for n in t.colnames:
        if len(n) < 3 :
            star_elements.append(n) #list of elements in table
    elements = np.array(star_elements)
    delnotemp=[]
    slopes = []

    for d, nam in tqdm(enumerate(elements)):
        delnotemp.append(delta_minimized(t, nam)[1])
        abundance_notemp = residuals(t['age'], t[nam], t[nam + '_err'])

        x0 = (.01, .03, .07) #initial guess
        resnotemp = scipy.optimize.minimize(twodnlnL, x0, args = (t['age'], abundance_notemp, t['age_err'], t[nam + '_err']))
    
        plt.ioff()
        fig, ax = plt.subplots()
    
        ax.scatter(t['age'], abundance_notemp, c='forestgreen') 
        ax.set_xlabel('Age',fontsize='xx-large')
        ax.set_ylabel(nam + '/Fe', fontsize='xx-large')
        ax.set_title(nam + ' Abundance vs. Stellar Age', fontsize='xx-large')

        #line of best fit
        mb = find_m_b(t['age'], abundance_notemp, t[nam + '_err'])
        slopes.append(mb[0])
        for i, txt in enumerate (t[nam]):
            plot_xs = np.arange(0, 9, .01)
            ax.plot(plot_xs, mb[0] * plot_xs + (mb[1]), color = 'palevioletred', linewidth=1)
    
       #point labels
        #for i, txt in enumerate(t['star_name']): 
                #ax.annotate(txt, xy=(t['age'][i], abundance_notemp[i]), xytext = (-5,5), fontsize='small', 
                    #textcoords='offset points', ha='center', va='bottom')
            
        mbtemp= resnotemp['x']
        plot_xs = np.arange(0, 9, .1)
        ax.plot(plot_xs, mbtemp[0] * plot_xs + (mbtemp[1]), color = 'black', linewidth=1)
        #ax.text(2,.15, deltemp[y] ,horizontalalignment='right', fontsize=12)
    
        fig.savefig(nam +'_age_no_temp.png')
        plt.close(fig)

def percent_diff(val1, val2):
    return np.absolute(((val1-val2)/((val1+ val2)/2)) *100)

def stdev(x):
    mean = np.mean(x)
    fun = 0
    for c in x: 
        j = (c-mean)**2
        fun = fun + j 
    s = np.sqrt(fun/(len(x)))
    return s

if __name__ == "__main__":
    mbvalues = []
    starrow = []
    x0 = [0,.1]
    #age_abund_plot(t)                                                                                                                                                                                  
    #age_abund_plot_no_temp(t) 

    for i, txt in enumerate(t['star_name']):
        tabl = star_table(txt)
        temperature= tabl.columns[3]
        elementab = tabl.columns[1]
        aberrors =tabl.columns[2]

    #mbvalues is an array (slope, intercept, standard deviation of slope & intercept, standard slope error, intercept error, chi squared)
        starrow.append(find_m_b(temperature,elementab, aberrors))
        jack = jackknifemb(np.array(temperature),np.array(elementab),np.array(aberrors))
        standarddev = (statistics.stdev(jack[0]),statistics.stdev(jack[1]))
        starrow.append(standarddev)
        errors = (standardslopeerror(temperature,elementab,aberrors), standardintercepterror(temperature,elementab,aberrors))
        starrow.append(errors)
        starrow.append(chisquared(x0, temperature, elementab, aberrors))
    
        mbvalues.append(starrow)
        starrow =[]
    
        plt.figure()
        
        #abudiff_noCO(txt)
        abund_plot_noCO(txt)
