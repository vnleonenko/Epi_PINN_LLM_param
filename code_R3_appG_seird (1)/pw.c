void seird_pw(int nb,const double*betas,const double*bps,double sigma,double gamma,double mu,double S,double E,double I,double R,double D,int n,double N,int sird,double*oS,double*oE,double*oI,double*oR,double*oD){
  const int spd=50; const double h=0.02,h2=0.01,h6=0.02/6.0; double gmu=gamma+mu;
  oS[0]=S;oE[0]=E;oI[0]=I;oR[0]=R;oD[0]=D; int seg=0;
  for(int day=0;day<n-1;day++){ for(int k=0;k<spd;k++){
    double t=day+k*h; while(seg<nb && t>=bps[seg]) seg++; double beta=betas[seg];
    double inf,dS1,dE1,dI1,dR1,dD1,dS2,dE2,dI2,dR2,dD2,dS3,dE3,dI3,dR3,dD3,dS4,dE4,dI4,dR4,dD4,S1,E1,I1,S2,E2,I2,S3,E3,I3;
    #define DERIV(Sx,Ex,Ix,dS,dE,dI,dR,dD) inf=beta*Sx*Ix/N; dS=-inf; if(sird){dE=0;dI=inf-gmu*Ix;} else {dE=inf-sigma*Ex; dI=sigma*Ex-gmu*Ix;} dR=gamma*Ix; dD=mu*Ix;
    DERIV(S,E,I,dS1,dE1,dI1,dR1,dD1)
    S1=S+h2*dS1;E1=E+h2*dE1;I1=I+h2*dI1;
    DERIV(S1,E1,I1,dS2,dE2,dI2,dR2,dD2)
    S2=S+h2*dS2;E2=E+h2*dE2;I2=I+h2*dI2;
    DERIV(S2,E2,I2,dS3,dE3,dI3,dR3,dD3)
    S3=S+h*dS3;E3=E+h*dE3;I3=I+h*dI3;
    DERIV(S3,E3,I3,dS4,dE4,dI4,dR4,dD4)
    S+=h6*(dS1+2*dS2+2*dS3+dS4);E+=h6*(dE1+2*dE2+2*dE3+dE4);I+=h6*(dI1+2*dI2+2*dI3+dI4);R+=h6*(dR1+2*dR2+2*dR3+dR4);D+=h6*(dD1+2*dD2+2*dD3+dD4);
  } oS[day+1]=S;oE[day+1]=E;oI[day+1]=I;oR[day+1]=R;oD[day+1]=D; }
}
