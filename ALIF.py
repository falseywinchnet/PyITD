#A. Cicone, J. Liu, H. Zhou. Adaptive Local Iterative Filtering for Signal Decomposition and Instantaneous Frequency analysis. Applied and Computational Harmonic Analysis, Volume 41, Issue 2, September 2016, Pages 384-411. doi:10.1016/j.acha.2016.03.001 Arxiv http://arxiv.org/abs/1411.6051

#warning: this is not python code. This is badly, badly transcribed matlab!
#you are encouraged to throw this away and start over from SCRATCH
#this code will NOT VALIDATE and will NOT RUN, it is unfinished, only a few indexes have been changed
#and only some notation has been fixed up. 
#crap that needs to be fixed just off the top:
#remove/decide what to do with strcmp instances
#replace matlab cell references
#all the TODO basically all the calls to ceil, floor, etc and make sure we have python equivalents that behave like matlab code
#throw out everything when i realize i already have code that will do most of this with a few tweaks

import numpy as np
import numpy
import decimal

def round_divmod(b,a):
    n,d = np.frompyfunc(lambda x:decimal.Decimal(x).as_integer_ratio(),1,2)(a.astype('U'))
    n,d = n.astype(int),d.astype(int)
    q,r = np.divmod(b*d,n)
    return q,r/d
#above function borrowed from stackoverflow, implements matlab-like mod(a,b)


def Maxmins_v2(f, extensiontype)
    if nargin == 1:
        extensionType = 'c'   
    N = length(f)
    Maxs=numpy.zeros(1,N)
    Mins=numpy.zeros(1,N)
    df = [ x-y for (x,y) in zip(x[1:],x[:-1]) ] #diff(f)
    h = 1
    cIn=0
    if strcmp(extensionType,'p') and df[1] == 0 and df[end] == 0:
        while df[h]==0:
            cIn=cIn+1
            h=h+1;
    if df[h] < 0:
        Initial_df=-1
    else:
        Initial_df=+1

    cmaxs=0
    cmins=0

    if strcmp(extensionType,'c') and df[1] == 0:
    while df[h]==0:
        h=h+1
    if df[h] < 0:
        cmaxs=cmaxs+1
        Maxs[cmaxs]=h
    else:
        cmins=cmins+1
        Mins[cmins]=h
    c = 0
    last_df=[]
    for i in range(h,N-2):
        if df[i]*df[i+1] == 0:
            if df[i] < 0:
                last_df=-1
                posc = i
            elif df[i] > 0:
                last_df=+1
                posc = i
            c = c + 1
            if df[i+1] < 0:
                if last_df==+1:
                    cmaxs=cmaxs+1
                    Maxs[cmaxs]=posc+floor((c-1)/2)+1                
                c=0
            if df[i+1] > 0:
                if last_df==-1:
                    cmins=cmins+1
                    Mins[cmins]=posc+floor((c-1)/2)+1             
                c=0
        if df[i]*df[i+1] < 0:
            if df[i] < df[i+1]:
                cmins=cmins+1
                Mins[cmins]=i+1
                last_df=-1
            else:
                cmaxs=cmaxs+1
                Maxs[cmaxs]=i+1
                last_df=+1

    if c > 0:
        if strcmp(extensionType,'p'):
            print('Code to be completed!')
            if Initial_df < 0:
                if last_df==+1:
                    cmaxs=cmaxs+1
                    Maxs[cmaxs]=mod(posc+floor((c+cIn-1)/2)+1,N)             
            elif Initial_df > 0:
                if last_df==+1:
                    cmins=cmins+1
                    Mins[cmins]=mod(posc+floor((c+cIn-1)/2)+1,N)                
            else:
                print('Code missing!')
        if strcmp(extensionType,'c'):        
            if last_df > 0:
                cmaxs=cmaxs+1
                Maxs[cmaxs]=posc+1
            else:
                cmins=cmins+1
                Mins[cmins]=posc+1
        if Mins[cmins]==0:
            Mins[cmins]=N
        if Maxs[cmaxs]==0:
            Maxs[cmaxs]=N

    Maxs=Maxs[0:cmaxs]
    Mins=Mins[0:cmins]
    maxmins=sort([Maxs Mins])#TODO

    if strcmp(extensionType,'p'): # we deal with a periodic signal
        print('Code to be completed')
        if isempty(maxmins):
            maxmins = 1
        else:
            if maxmins[1]~=1 and maxmins[end]~=N:
                if (f[maxmins[end]] > f[maxmins[end]+1]\
                    and f[maxmins[1]] > f[maxmins[1]-1]) or (f[maxmins[end]] < f[maxmins[end]+1]\
                                                            and f[maxmins[1]] < f[maxmins[1]-1]):
                    maxmins=[1 maxmins]

    elif strcmp(extensionType,'c'):
        if not(isempty(maxmins)):
            if maxmins[1] ~= 1 and maxmins[end] ~= N and df[1]~=0 and df[end]~=0:
                if Maxs[1] < Mins[1]:
                    Mins=[1 Mins]
                else:
                    Maxs=[1 Maxs]
                if Maxs[end] < Mins[end]:
                    Maxs=[Maxs N]
                else:
                    Mins=[Mins N]
                maxmins = [1, maxmins, N]
            elif maxmins[1] ~= 1 and df[1]~=0:
                maxmins = [1, maxmins]
                if Maxs[1] < Mins[1]:
                    Mins=[1 Mins]
                else:
                    Maxs=[1 Maxs]
            elif  maxmins[end] ~= N and df[end]~=0:
                maxmins = [maxmins, N];
                if Maxs[end] < Mins[end]:
                    Maxs=[Maxs N]
                else:
                    Mins=[Mins N]
    elif strcmp(extensionType,'r'):
        #disp('Code to be completed')
        if isempty(maxmins):
            maxmins = [1, N]
        else:
            if maxmins[1] ~= f[1] and maxmins[end] ~= f[end]:
                maxmins = [1, maxmins, N]
            elif f(maxmins[1]) ~= f[1]:
                maxmins = [1, maxmins]
            elif  f[maxmins[end]] ~= f[end]:
                maxmins = [maxmins, N]

    if nargout<=1:
        varargout{1}=maxmins
    elif nargout==2:
        varargout{1}=Maxs
        varargout{2}=Mins
    return vargout



def get_mask_v1(y,k):
    # get the mask with length 2*k+1
    # k could be an integer or not an integer
    # y is the area under the curve for each bar
    n=len(y)
    m=(n-1)/2
    if k<=m: # The prefixed filter contains enough points
        if round_divmod(k,1)==0:     # if the mask_length is an integer
                                        #note: THIS WAS MATLAB MOD(k,1)
            a=numpy.zeros(1,2*k+1)
            for i=1:2*k+1:
                s=(i-1)*(2*m+1)/(2*k+1)+1
                t=i*(2*m+1)/(2*k+1)
            
                #s1=s-floor(s);
                s2=ceil(s)-s
            
                t1=t-floor(t)
                #t2=ceil(t)-t
            
                if floor(t)<1:
                    print('Ops')
                a(i)=sum(y(ceil(s):floor(t)))+s2*y(ceil(s))+t1*y(floor(t))
        
        else:  # if the mask length is not an integer
            new_k=floor(k) #TODO
            extra = k-new_k
            c=(2*m+1)/(2*new_k+1+2*extra)
        
            a=zeros(1,2*new_k+3)
        
            t=extra*c+1
            t1=t-floor(t)
            %t2=ceil(t)-t #TODO
            if k<0:
            print('Ops')
            a(1)=sum(y(1:floor(t)))+t1*y(floor(t)) #TODO
        
            for i=2:2*new_k+2:
                s=extra*c+(i-2)*c+1
                t=extra*c+(i-1)*c
                %s1=s-floor(s)
                s2=ceil(s)-s
            
                t1=t-floor(t)
            
                a(i)=math.fsum(y(ceil(s):floor(t)))+s2*y(ceil(s))+t1*y(floor(t)) #TODO
                
            t2=ceil(t)-t #TODO
            a(2*new_k+3)=math.fsum(y(ceil(t):n))+t2*y(ceil(t)) #TODO
    else: # We need a filter with more points than MM, we use interpolation
        dx=0.01
    # we assume that MM has a dx = 0.01, if m = 6200 it correspond to a
    # filter of length 62*2 in the physical space
        f=y/dx # function we need to interpolate
        dy=m*dx/k
        b=scipy.interp1d(0:m,f(m+1:2*m+1),0:m/k:m) #TODO
        if size(b,1)>size(b,2):
            b=b.'
    
        if size(b,1)>1:
            print('\n\nError!')
            print('The provided mask is not a vector!!')
            a=[]
            return
        a=[fliplr(b(2:end)) b]*dy
            #if abs(norm(a,1)-1)>10^-14
            #         fprintf('\n\nError\n\n')
            #         fprintf('Area under the mask = %2.20f\n',norm(a,1))
            #         fprintf('it should be equal to 1\nWe rescale it using its norm 1\n\n')
            a=a/norm(a,1) #TODO
    #####################
    return a





def alif(f,options):
    
    if nargin == 0,  help ALIFv5_4; return; end
    if nargin == 1, options = Settings_ALIF; end

    extensionType = 'p'; % used in the calculations of mins and maxs

    load('  ','MM');
    if len(f)>len(MM):
        print(['\n\n      ********   Warning  *********\n\n'...
        ' The filter MM should contain more points\n'...
        ' to properly decompose the given signal\n\n'...
        ' We will use interpolation to generate a\n'...
        ' filter with the proper number of points\n\n'...
        '      *****************************\n\n'])


    maxmins_f=Maxmins_v3(f,extensionType);


    while  len(maxmins_f) > (options.ALIF.ExtPoints) and size(IMF,1) < (options.ALIF.NIMFs):
    %% Outer loop
   
        h = f
    

    
        T_f=[diff(maxmins_f) (maxmins_f[1]+N-maxmins_f[end])]
        temp_T_f=[T_f T_f T_f T_f T_f T_f T_f T_f T_f T_f T_f]
        temp_maxmins_f=[maxmins_f maxmins_f+N maxmins_f+2*N maxmins_f+3*N \
            maxmins_f+4*N maxmins_f+5*N maxmins_f+6*N \
            maxmins_f+7*N maxmins_f+8*N maxmins_f+9*N maxmins_f+10*N]
        temp_iT_f= scipy.interp1(temp_maxmins_f,temp_T_f,1:11*N,'cubic')
        iT_f = temp_iT_f[5*N+1:6*N]
    
    
        nTry=1
    
        iT_f0=iT_f
    
        OK=0
        while OK==0:
            opts=Settings_IF_v1('IF.ExtPoints',3,'IF.NIMFs',nTry,'verbose',options.verbose,'IF.alpha',1)
            IMF_iT_f = IF_v8_3[iT_f0,opts]  # We use IF algo for periodic signals to compute the mask length
            if 0>=min(IMF_iT_f[end,:]) and (size(IMF_iT_f,1)-1)==nTry:
                nTry=nTry+1
            elif 0>=min(IMF_iT_f[end,:]) and not((size(IMF_iT_f,1)-1)==nTry):
                print('Negative mask length')
                return
            else:
                OK=1
        iT_f = IMF_iT_f[end,:]
    
        nn = 2*options.ALIF.xi
    
        iT_f = nn*iT_f
    
        if ceil(max(iT_f))>=floor(N/2): #TODO
            print('The computation of the IMF requires a Mask length bigger than the signal itself')
            print('From this IMF on you can try reducing the value of the paramter ALIF.Xi in Settings_ALIF')
            break
        else:

        mask_lengths(size(IMF,1)+1,:)=iT_f
     
        inStepN=0
     
        W=numpy.zeros(N,N)
        for i in range(0, N):     
            wn = get_mask_v1(MM, iT_f[i])
            wn=wn/norm(wn,1)
            length = (len(wn)-1)/2
            wn = [reshape(wn,1,2*len+1) numpy.zeros(1,N-2*length-1)];
        SD=Inf;
    
        while SD > options.ALIF.delta and inStepN<500:
        
            inStepN=inStepN+1
        
            ave = W*h'
                
            SD = norm(ave)^2/norm(h)^2;
            h = h - ave'
    
        IMF=[IMF; h]
    
        f = f-h
    
        maxmins_f=Maxmins_v3(f,extensionType)


    IMF =[IMF; f]

    if options.saveEnd == 1:
        save('Decomp_ALIF_v5_4.mat')
    ##################################
    return IMF, mask_lengths



