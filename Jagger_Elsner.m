function EM = Jagger_Elsner(H,priors)
 
    % This function returns the estimated number of events per season on X 
    % using a generilez linear model and atm. patterns as priors. This
    % model is explained in Jagger and Elsner 2012. 
    % DOI: https://doi.org/10.1175/JAMC-D-11-0107.1

    % The model is divided in two parts: 
    % 1) Cluster model: Do we have clusters (or not). 
    % 2) Hurricane model: How many events on each cluster. 

    % priors must be an array of [prior1, prior2] of size: # of seasons x # of priors
    %--------------------------------------------------------------------------
    [a b] = size(priors);
    %---Cluster model 
    I = H; I(I>0)=1;
    Mdl = fitglm(priors,I,"Distribution","binomial",Link="comploglog"); 
    
    int = NaN(length(H),b+1); 
    int(:,1) =  repmat(Mdl.Coefficients.Estimate(1),length(H),1); 
    for i = 2:b+1
    int(:,i) = Mdl.Coefficients.Estimate(i)*priors(:,i-1);
    end  
    r = exp(sum(int,2)); clear int; clear Mdl;
    
    %---Hurricane model 
    Mdl = fitglm(priors,H,"Distribution","poisson",Link="log");
    
    int = NaN(length(H),b+1); 
    int(:,1) =  repmat(Mdl.Coefficients.Estimate(1),length(H),1); 
    for i = 2:b+1
    int(:,i) = Mdl.Coefficients.Estimate(i)*priors(:,i-1);
    end  
    lambda = exp(sum(int,2));

    %---Find p value (slope)
    p = (lambda./r)-1;
    p = mean(p);

    %--------------------------------------------------------------------------
    %--- 3) Probability mass function
    nbins = length(0:1:max(unique(H)));
    PP = NaN(nbins,1);

    for k = 1:nbins-1
        for i = 0:1:floor(k/2)
          
            dpois = poisspdf(k-i,r);
            dbinom = binopdf(i,k-i,p);
        
            vb(i+1,:) = dpois.*dbinom;    
        end
        PP(k+1,1) = nansum(nansum(vb,1)); clear vb;
    end

    %---Special case k=0
    PP(1,1) = nansum(exp(-r));

    %---Estimated value
    EM = round((PP./sum(PP))*length(H));

end