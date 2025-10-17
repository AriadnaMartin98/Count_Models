function EM = HNB(H)

    % Hurdle Negative Binomial model:
    % This function returns the estimated number of events per season on X 
    % seasons based on obserevations of previus years, it allows overdispersion.

    % The model is divided in two parts:  H = S + C 
    % 1) Event-free seasons (or zero-observations): Do we have events (any) or no. Binary component. 
    % 2) Non-zero component: For no-free event seasons, how many events. Overdispersed component.


    %--------------------------------------------------------------------------
    %--- 1) S model: event-free seasons

    I = H; I(I>0)=1; %--- reduced dataset 

    %---Binomial distribution: since it 1: events, 0 no events
    S = sum(I);
    lambda = binofit(S,length(H));

    %--------------------------------------------------------------------------
    %--- 2) C model: clustering
    %--- When I == 1; means is not an event-free season, lets see if it has
    %cluster. For that we use the Negative Binomial. 
    %which can account for overdispersion

    H_nan = H; H_nan(H_nan==0) = NaN; %---reduced dataset

    H_nan_shifted = H_nan - 1; %Shift the H's so that 1 is now zero as distribution fits from 0 to inf

    param= mle(H_nan_shifted,'Distribution','Negative Binomial');

    %--------------------------------------------------------------------------
    %--- Avoid problems with singularities
     
    [warnmsg, msgid] = lastwarn;

    if strcmp(msgid,'MATLAB:nearlySingularMatrix')

        EM = NaN;
        lastwarn('')
       
        return

    end

    %--------------------------------------------------------------------------
    %--- 3) Probability mass function
    nbins = length(0:1:max(unique(H)));
    PP = NaN(nbins,1);

    for k = 1:max(H)
          
        dS = lambda;
        
        %k-1 as data has been shifted e.g. number of hurricanes equals H_nan + 1
        dC = nbinpdf(k-1,param(1),param(2));
        
        PP(k+1) = dS.*dC;   
    end

    %--- Special case k = 0
    PP(1) =  1-lambda;
%     disp(sum(PP));

%     %--------------------------------------------------------------------------
%     %--- Test Converge
%     k = 0:20;        % Extended range for successes
%     y = nbinpdf(k, param(1), param(2));
% 
%     total_prob = sum(y); % Sum of probabilities
%     disp(total_prob); % Close to 1 for a sufficiently large range
%     %--------------------------------------------------------------------------
   
    %--------------------------------------------------------------------------
    %---Estimated value
    EM = round((PP./nansum(PP)).*length(H));
    %--------------------------------------------------------------------------

end





