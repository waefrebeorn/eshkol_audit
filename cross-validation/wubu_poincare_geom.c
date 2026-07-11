/*
 * wubu_poincare_geom.c -- implementation (see wubu_poincare_geom.h).
 * Standalone port of tsotchke/eshkol lib/core/manifold.esk geometry (MIT).
 */
#include "wubu_poincare_geom.h"

void poincare_exp_eshkol(float*o,const float*p,const float*v,int n,float c){
    if(c<=0){wpg_vadd(o,p,v,n);return;}
    float vn=wpg_vnorm(v,n);
    if(vn<WPG_EPS){memcpy(o,p,n*sizeof(float));return;}
    float p2=wpg_vdot(p,p,n);
    float lam=2.0f/(1.0f-c*p2+WPG_EPS);
    float t=tanhf(0.5f*lam*vn);
    float factor=t/(vn+WPG_EPS);
    float s[64]; wpg_vscale(s,v,factor,n);
    wpg_mobius_add(o,p,s,n,c);
}

void poincare_exp_correct(float*o,const float*p,const float*v,int n,float c){
    if(c<=0){wpg_vadd(o,p,v,n);return;}
    float vn=wpg_vnorm(v,n);
    if(vn<WPG_EPS){memcpy(o,p,n*sizeof(float));return;}
    float p2=wpg_vdot(p,p,n);
    float lam=2.0f/(1.0f-c*p2+WPG_EPS);
    float arg=0.5f*sqrtf(c)*vn;
    float t=tanhf(arg);
    float factor=lam*t/(sqrtf(c)*vn+WPG_EPS);
    float s[64]; wpg_vscale(s,v,factor,n);
    wpg_mobius_add(o,p,s,n,c);
}

float eshkol_distance(const float*a,const float*b,int n,float c){
    float d2=0,na=0,nb=0;
    for(int i=0;i<n;i++){float dd=a[i]-b[i];d2+=dd*dd;na+=a[i]*a[i];nb+=b[i]*b[i];}
    float arg=1.0f+2.0f*c*d2/(((1.0f-c*na)+WPG_EPS)*(1.0f-c*nb)+WPG_EPS);
    return acoshf(fmaxf(1.0f,arg));
}

float manifold_lambda(ManifoldType t,const float*x,int n,float c){
    float nx2=wpg_vdot(x,x,n);
    if(t==MAN_HYPERBOLIC) return 2.0f/(1.0f-c*nx2+WPG_EPS);
    if(t==MAN_SPHERICAL)   return 2.0f/(1.0f+c*nx2+WPG_EPS);
    return 1.0f;
}
float manifold_christoffel(ManifoldType t,const float*x,int n,int i,int j,int k,float c){
    float lam=manifold_lambda(t,x,n,c);
    if(t==MAN_HYPERBOLIC){
        if(i==j) return -c*lam*x[k];
        if(i==k) return  c*lam*x[j];
        if(j==k) return  c*lam*x[i];
        return 0.0f;
    }
    if(t==MAN_SPHERICAL){
        if(i==j) return  c*lam*x[k];
        if(i==k) return -c*lam*x[j];
        if(j==k) return -c*lam*x[i];
        return 0.0f;
    }
    return 0.0f;
}
float manifold_sectional_curvature(ManifoldType t,float c){
    if(t==MAN_HYPERBOLIC) return -c;
    if(t==MAN_SPHERICAL)   return  c;
    return 0.0f;
}
float manifold_scalar_curvature(ManifoldType t,int n,float c){
    return manifold_sectional_curvature(t,c)*(float)n*(float)(n-1);
}
float manifold_ricci(ManifoldType t,const float*x,int n,int i,int j,float c){
    float lam=manifold_lambda(t,x,n,c);
    return manifold_sectional_curvature(t,c)*(float)(n-1)*lam*lam*(i==j?1.0f:0.0f);
}
