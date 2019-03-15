% CDF MATCHING polynomial
function [SATCDF_ok,COEFF]=QMAPP(OBS,SAT)
load dataSM
D_OBS = OBS(:,1);
OBSdata=(OBS(:,2));

D_SAT = SAT(:,1);
SATdata=(SAT(:,2));
SATCDF_ok = NaN(length(SATdata),1);

for i=1:12     % for each month of the year
    OBSmont = OBSdata(find(month(D_OBS)==i));
    ID_SATmont = find(month(D_SAT)==i); SATmont = SATdata(ID_SATmont);
    
    POBS = [1:length(OBSmont)]'./(length(OBSmont)+1);
    PSAT = [1:length(SATmont)]'./(length(SATmont)+1);
    SATint= interp1(PSAT,sort(SATmont),sort(POBS),'linear','extrap');
    
    
    DIFF=sort(OBSmont)-SATint;
    COEFF= polyfit(SATint,DIFF, 5);
    SATCDF= polyval(COEFF,SATmont)+SATmont;
    %     SATCDF(SATCDF> max(SATdata))=NaN;
    %     SATCDF(SATCDF< 0)=0;    % setting to zero negative data
    SATCDF_ok(ID_SATmont) = SATCDF;
    
    
    subplot(3,4,i)
    plot( sort(OBSmont),(1:length(OBSmont))/(length(OBSmont)+1),'Color',0.7*[1,1,1], 'linewidth',7)
    hold on
    plot(sort(SATmont),(1:length(SATmont))/(length(SATmont)+1), 'b-','linewidth',4)
    plot( sort(SATCDF),(1:length(SATCDF))/(length(SATCDF)+1), 'r--', 'linewidth',2)
    xlabel('data'), ylabel('probability density function')
    if i==12, legend ('Reference data','Original biased data','Corrected data',4), end
    
    
    M_STAT_OBS(i,1)= nanmean(OBSmont); V_STAT_OBS(i,1) = nanvar(OBSmont);
    M_STAT_ST(i,1)= nanmean(SATCDF);   V_STAT_ST(i,1) = nanvar(SATCDF);
    
end
