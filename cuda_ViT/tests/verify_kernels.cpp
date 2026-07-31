// Emulador CPU de los kernels tileados: replica LITERALMENTE las expresiones de
// indice de gemm.cu y attention.cu (misma estructura de bloques/hilos/tiles) y
// compara contra una referencia ingenua. Sirve para validar la indexacion sin GPU.
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

#define TILE 16
static int ceil_div(int a,int b){return (a+b-1)/b;}

static double maxdiff(const std::vector<float>&a,const std::vector<float>&b){
    double m=0; for(size_t i=0;i<a.size();++i) m=std::max(m,(double)std::fabs(a[i]-b[i])); return m;
}
static int fails=0;
static void check(const char*name,double d,double tol=1e-4){
    bool ok = d<=tol && !std::isnan(d);
    printf("  %-38s max|dif| = %.3e   %s\n", name, d, ok?"OK":"<<<< FALLA");
    if(!ok) fails++;
}

// ---------------------------------------------------------------- referencias
static void ref_nn(const float*A,const float*B,const float*bias,float*C,int M,int N,int K){
    for(int m=0;m<M;++m)for(int n=0;n<N;++n){double s=0;for(int k=0;k<K;++k)s+=(double)A[m*K+k]*B[k*N+n];
        C[m*N+n]=(float)s+(bias?bias[n]:0.f);}
}
static void ref_nt(const float*A,const float*B,float*C,int M,int N,int K){
    for(int m=0;m<M;++m)for(int n=0;n<N;++n){double s=0;for(int k=0;k<K;++k)s+=(double)A[m*K+k]*B[n*K+k];C[m*N+n]=(float)s;}
}
static void ref_tn(const float*A,const float*B,float*C,int M,int N,int K){
    for(int m=0;m<M;++m)for(int n=0;n<N;++n){double s=0;for(int k=0;k<K;++k)s+=(double)A[k*M+m]*B[k*N+n];C[m*N+n]=(float)s;}
}

// ------------------------------------------------- emulacion de gemm_nn_kernel
static void emu_gemm_nn(const float*A,const float*B,const float*bias,float*C,int M,int N,int K){
    for(int by=0;by<ceil_div(M,TILE);++by)for(int bx=0;bx<ceil_div(N,TILE);++bx){
        float As[TILE][TILE], Bs[TILE][TILE+1], acc[TILE][TILE]={};
        for(int t=0;t<K;t+=TILE){
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
                int row=by*TILE+ty, col=bx*TILE+tx;
                int a_col=t+tx, b_row=t+ty;
                As[ty][tx]=(row<M&&a_col<K)?A[row*K+a_col]:0.f;
                Bs[ty][tx]=(b_row<K&&col<N)?B[b_row*N+col]:0.f;
            }
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx)
                for(int i=0;i<TILE;++i) acc[ty][tx]+=As[ty][i]*Bs[i][tx];
        }
        for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
            int row=by*TILE+ty,col=bx*TILE+tx;
            if(row<M&&col<N) C[row*N+col]=acc[ty][tx]+(bias?bias[col]:0.f);
        }
    }
}

// ------------------------------------------------- emulacion de gemm_nt_kernel
static void emu_gemm_nt(const float*A,const float*B,float*C,int M,int N,int K){
    for(int by=0;by<ceil_div(M,TILE);++by)for(int bx=0;bx<ceil_div(N,TILE);++bx){
        float As[TILE][TILE+1],Bs[TILE][TILE+1],acc[TILE][TILE]={};
        for(int t=0;t<K;t+=TILE){
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
                int row=by*TILE+ty;
                int a_col=t+tx;
                As[ty][tx]=(row<M&&a_col<K)?A[row*K+a_col]:0.f;
                int b_row=bx*TILE+ty, b_col=t+tx;
                Bs[ty][tx]=(b_row<N&&b_col<K)?B[b_row*K+b_col]:0.f;
            }
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx)
                for(int i=0;i<TILE;++i) acc[ty][tx]+=As[ty][i]*Bs[tx][i];
        }
        for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
            int row=by*TILE+ty,col=bx*TILE+tx;
            if(row<M&&col<N) C[row*N+col]=acc[ty][tx];
        }
    }
}

// ------------------------------------------------- emulacion de gemm_tn_kernel
static void emu_gemm_tn(const float*A,const float*B,float*C,int M,int N,int K){
    for(int by=0;by<ceil_div(M,TILE);++by)for(int bx=0;bx<ceil_div(N,TILE);++bx){
        float As[TILE][TILE+1],Bs[TILE][TILE+1],acc[TILE][TILE]={};
        for(int t=0;t<K;t+=TILE){
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
                int a_row=t+ty;
                As[ty][tx]=(a_row<K&&(by*TILE+tx)<M)?A[a_row*M+by*TILE+tx]:0.f;
                int b_row=t+ty, col=bx*TILE+tx;
                Bs[ty][tx]=(b_row<K&&col<N)?B[b_row*N+col]:0.f;
            }
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx)
                for(int i=0;i<TILE;++i) acc[ty][tx]+=As[i][ty]*Bs[i][tx];
        }
        for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
            int row=by*TILE+ty,col=bx*TILE+tx;
            if(row<M&&col<N) C[row*N+col]=acc[ty][tx];
        }
    }
}

// ------------------------------- emulacion de los bmm_* (global y shared)
static void emu_bmm_nt_global(const float*A,const float*B,float*C,int bt,int M,int N,int K,
                              int sA,int sB,int sC,float alpha){
    for(int bh=0;bh<bt;++bh)for(int row=0;row<M;++row)for(int col=0;col<N;++col){
        const float*a=A+(size_t)bh*sA+(size_t)row*K, *b=B+(size_t)bh*sB+(size_t)col*K;
        float acc=0; for(int i=0;i<K;++i)acc+=a[i]*b[i];
        C[(size_t)bh*sC+(size_t)row*N+col]=alpha*acc;
    }
}
static void emu_bmm_nt_shared(const float*A,const float*B,float*C,int bt,int M,int N,int K,
                              int sA,int sB,int sC,float alpha){
    for(int bh=0;bh<bt;++bh)for(int by=0;by<ceil_div(M,TILE);++by)for(int bx=0;bx<ceil_div(N,TILE);++bx){
        float As[TILE][TILE+1],Bs[TILE][TILE+1],acc[TILE][TILE]={};
        const float*a=A+(size_t)bh*sA,*b=B+(size_t)bh*sB;
        for(int t=0;t<K;t+=TILE){
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
                int row=by*TILE+ty, ak=t+tx;
                As[ty][tx]=(row<M&&ak<K)?a[(size_t)row*K+ak]:0.f;
                int brow=bx*TILE+ty, bk=t+tx;
                Bs[ty][tx]=(brow<N&&bk<K)?b[(size_t)brow*K+bk]:0.f;
            }
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx)
                for(int i=0;i<TILE;++i)acc[ty][tx]+=As[ty][i]*Bs[tx][i];
        }
        for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
            int row=by*TILE+ty,col=bx*TILE+tx;
            if(row<M&&col<N)C[(size_t)bh*sC+(size_t)row*N+col]=alpha*acc[ty][tx];
        }
    }
}
static void emu_bmm_nn_global(const float*A,const float*B,float*C,int bt,int M,int N,int K,
                              int sA,int sB,int sC,float alpha){
    for(int bh=0;bh<bt;++bh)for(int row=0;row<M;++row)for(int col=0;col<N;++col){
        const float*a=A+(size_t)bh*sA+(size_t)row*K,*b=B+(size_t)bh*sB;
        float acc=0;for(int i=0;i<K;++i)acc+=a[i]*b[(size_t)i*N+col];
        C[(size_t)bh*sC+(size_t)row*N+col]=alpha*acc;
    }
}
static void emu_bmm_nn_shared(const float*A,const float*B,float*C,int bt,int M,int N,int K,
                              int sA,int sB,int sC,float alpha){
    for(int bh=0;bh<bt;++bh)for(int by=0;by<ceil_div(M,TILE);++by)for(int bx=0;bx<ceil_div(N,TILE);++bx){
        float As[TILE][TILE+1],Bs[TILE][TILE+1],acc[TILE][TILE]={};
        const float*a=A+(size_t)bh*sA,*b=B+(size_t)bh*sB;
        for(int t=0;t<K;t+=TILE){
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
                int row=by*TILE+ty,col=bx*TILE+tx,ak=t+tx,bk=t+ty;
                As[ty][tx]=(row<M&&ak<K)?a[(size_t)row*K+ak]:0.f;
                Bs[ty][tx]=(bk<K&&col<N)?b[(size_t)bk*N+col]:0.f;
            }
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx)
                for(int i=0;i<TILE;++i)acc[ty][tx]+=As[ty][i]*Bs[i][tx];
        }
        for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
            int row=by*TILE+ty,col=bx*TILE+tx;
            if(row<M&&col<N)C[(size_t)bh*sC+(size_t)row*N+col]=alpha*acc[ty][tx];
        }
    }
}
static void emu_bmm_tn_global(const float*A,const float*B,float*C,int bt,int M,int N,int K,
                              int sA,int sB,int sC,float alpha){
    for(int bh=0;bh<bt;++bh)for(int row=0;row<M;++row)for(int col=0;col<N;++col){
        const float*a=A+(size_t)bh*sA,*b=B+(size_t)bh*sB;
        float acc=0;for(int i=0;i<K;++i)acc+=a[(size_t)i*M+row]*b[(size_t)i*N+col];
        C[(size_t)bh*sC+(size_t)row*N+col]=alpha*acc;
    }
}
static void emu_bmm_tn_shared(const float*A,const float*B,float*C,int bt,int M,int N,int K,
                              int sA,int sB,int sC,float alpha){
    for(int bh=0;bh<bt;++bh)for(int by=0;by<ceil_div(M,TILE);++by)for(int bx=0;bx<ceil_div(N,TILE);++bx){
        float As[TILE][TILE+1],Bs[TILE][TILE+1],acc[TILE][TILE]={};
        const float*a=A+(size_t)bh*sA,*b=B+(size_t)bh*sB;
        for(int t=0;t<K;t+=TILE){
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
                int ak=t+ty, am=by*TILE+tx;
                As[ty][tx]=(ak<K&&am<M)?a[(size_t)ak*M+am]:0.f;
                int bk=t+ty,col=bx*TILE+tx;
                Bs[ty][tx]=(bk<K&&col<N)?b[(size_t)bk*N+col]:0.f;
            }
            for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx)
                for(int i=0;i<TILE;++i)acc[ty][tx]+=As[i][ty]*Bs[i][tx];
        }
        for(int ty=0;ty<TILE;++ty)for(int tx=0;tx<TILE;++tx){
            int row=by*TILE+ty,col=bx*TILE+tx;
            if(row<M&&col<N)C[(size_t)bh*sC+(size_t)row*N+col]=alpha*acc[ty][tx];
        }
    }
}

// -------------------------------------------- reordenamientos de attention.cu
static void emu_split_qkv(const float*qkv,float*q,float*k,float*v,int B,int T,int H,int Dh){
    int D=H*Dh;
    for(int idx=0;idx<B*T*D;++idx){
        int d=idx%Dh,h=(idx/Dh)%H,t=(idx/D)%T,b=idx/(T*D);
        size_t src=((size_t)b*T+t)*(3*D)+(size_t)h*Dh+d;
        size_t dst=(((size_t)b*H+h)*T+t)*Dh+d;
        q[dst]=qkv[src];k[dst]=qkv[src+D];v[dst]=qkv[src+2*D];
    }
}
static void emu_merge_qkv_grad(const float*dq,const float*dk,const float*dv,float*dqkv,int B,int T,int H,int Dh){
    int D=H*Dh;
    for(int idx=0;idx<B*T*D;++idx){
        int d=idx%Dh,h=(idx/Dh)%H,t=(idx/D)%T,b=idx/(T*D);
        size_t dst=((size_t)b*T+t)*(3*D)+(size_t)h*Dh+d;
        size_t src=(((size_t)b*H+h)*T+t)*Dh+d;
        dqkv[dst]=dq[src];dqkv[dst+D]=dk[src];dqkv[dst+2*D]=dv[src];
    }
}
static void emu_merge_heads(const float*attn,float*y,int B,int T,int H,int Dh){
    int D=H*Dh;
    for(int idx=0;idx<B*T*D;++idx){
        int d=idx%Dh,h=(idx/Dh)%H,t=(idx/D)%T,b=idx/(T*D);
        y[((size_t)b*T+t)*D+h*Dh+d]=attn[(((size_t)b*H+h)*T+t)*Dh+d];
    }
}
static void emu_split_heads(const float*dy,float*da,int B,int T,int H,int Dh){
    int D=H*Dh;
    for(int idx=0;idx<B*T*D;++idx){
        int d=idx%Dh,h=(idx/Dh)%H,t=(idx/D)%T,b=idx/(T*D);
        da[(((size_t)b*H+h)*T+t)*Dh+d]=dy[((size_t)b*T+t)*D+h*Dh+d];
    }
}

// ---------------------------------------------------- im2patch de patch_embed.cu
static void emu_im2patch(const float*img,float*patches,int B,int img_size,int patch){
    int gs=img_size/patch, pd=patch*patch, np=gs*gs;
    for(int idx=0;idx<B*np*pd;++idx){
        int k=idx%pd,p=(idx/pd)%np,b=idx/(pd*np);
        int pr=p/gs,pc=p%gs,kr=k/patch,kc=k%patch;
        int r=pr*patch+kr,c=pc*patch+kc;
        patches[idx]=img[((size_t)b*img_size+r)*img_size+c];
    }
}

// ============================ chequeos de gradiente (formulas analiticas) =====
// LayerNorm
static void ln_fwd(const float*x,const float*g,const float*b,float*y,float*mean,float*rstd,int R,int D,float eps){
    for(int i=0;i<R;++i){
        double mu=0;for(int j=0;j<D;++j)mu+=x[i*D+j];mu/=D;
        double var=0;for(int j=0;j<D;++j){double d=x[i*D+j]-mu;var+=d*d;}var/=D;
        float rs=1.f/std::sqrt((float)var+eps);
        mean[i]=(float)mu;rstd[i]=rs;
        for(int j=0;j<D;++j)y[i*D+j]=g[j]*((x[i*D+j]-(float)mu)*rs)+b[j];
    }
}
static void ln_bwd(const float*dy,const float*x,const float*g,const float*mean,const float*rstd,
                   float*dx,float*dg,float*db,int R,int D){
    for(int j=0;j<D;++j){dg[j]=0;db[j]=0;}
    for(int i=0;i<R;++i){
        double m1=0,m2=0;
        for(int j=0;j<D;++j){
            float xhat=(x[i*D+j]-mean[i])*rstd[i];
            float dxh=dy[i*D+j]*g[j];
            m1+=dxh;m2+=dxh*xhat;
        }
        m1/=D;m2/=D;
        for(int j=0;j<D;++j){
            float xhat=(x[i*D+j]-mean[i])*rstd[i];
            float dxh=dy[i*D+j]*g[j];
            dx[i*D+j]=rstd[i]*(dxh-(float)m1-xhat*(float)m2);
            dg[j]+=dy[i*D+j]*xhat;
            db[j]+=dy[i*D+j];
        }
    }
}
// Softmax
static void sm_fwd(const float*x,float*y,int R,int C){
    for(int i=0;i<R;++i){
        float mx=-1e30f;for(int j=0;j<C;++j)mx=std::max(mx,x[i*C+j]);
        double s=0;for(int j=0;j<C;++j){float e=std::exp(x[i*C+j]-mx);y[i*C+j]=e;s+=e;}
        for(int j=0;j<C;++j)y[i*C+j]/=(float)s;
    }
}
static void sm_bwd(const float*P,const float*dP,float*dS,int R,int C,float scale){
    for(int i=0;i<R;++i){
        double dot=0;for(int j=0;j<C;++j)dot+=(double)P[i*C+j]*dP[i*C+j];
        for(int j=0;j<C;++j)dS[i*C+j]=scale*P[i*C+j]*(dP[i*C+j]-(float)dot);
    }
}
// GELU
static float gelu(float v){const float c=0.7978845608028654f,a=0.044715f;
    return 0.5f*v*(1.f+std::tanh(c*(v+a*v*v*v)));}
static float gelu_d(float v){const float c=0.7978845608028654f,a=0.044715f;
    float u=c*(v+a*v*v*v),t=std::tanh(u),du=c*(1.f+3.f*a*v*v);
    return 0.5f*(1.f+t)+0.5f*v*(1.f-t*t)*du;}

int main(){
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f,1.f);
    auto rnd=[&](int n){std::vector<float>v(n);for(auto&x:v)x=nd(rng);return v;};

    printf("\n=== 1. GEMM tileados (capas lineales) ===\n");
    // Se prueban tamanos NO multiplos de 16 a proposito, para ejercitar las guardas.
    int cases[][3]={{64,64,64},{50,37,23},{3200,192,64},{17,10,64},{5,128,64}};
    for(auto&cs:cases){
        int M=cs[0],N=cs[1],K=cs[2];
        auto A=rnd(M*K),Bm=rnd(K*N),bias=rnd(N);
        std::vector<float> C1(M*N),C2(M*N);
        ref_nn(A.data(),Bm.data(),bias.data(),C1.data(),M,N,K);
        emu_gemm_nn(A.data(),Bm.data(),bias.data(),C2.data(),M,N,K);
        char nm[96];snprintf(nm,sizeof nm,"gemm_nn  M=%d N=%d K=%d",M,N,K);
        check(nm,maxdiff(C1,C2),1e-3);

        auto Bt=rnd(N*K);
        ref_nt(A.data(),Bt.data(),C1.data(),M,N,K);
        emu_gemm_nt(A.data(),Bt.data(),C2.data(),M,N,K);
        snprintf(nm,sizeof nm,"gemm_nt  M=%d N=%d K=%d",M,N,K);
        check(nm,maxdiff(C1,C2),1e-3);

        auto At=rnd(K*M);
        ref_tn(At.data(),Bm.data(),C1.data(),M,N,K);
        emu_gemm_tn(At.data(),Bm.data(),C2.data(),M,N,K);
        snprintf(nm,sizeof nm,"gemm_tn  M=%d N=%d K=%d",M,N,K);
        check(nm,maxdiff(C1,C2),1e-3);
    }

    printf("\n=== 2. BMM de atencion: global vs shared vs referencia ===\n");
    for(int patch : {4,7,14}){
        int np=(28/patch)*(28/patch), T=np+1, Dh=16, BH=8;
        // QK^T : bmm_nt
        {
            auto q=rnd(BH*T*Dh),k=rnd(BH*T*Dh);
            std::vector<float> R(BH*T*T),G(BH*T*T),S(BH*T*T);
            for(int b=0;b<BH;++b) ref_nt(q.data()+b*T*Dh,k.data()+b*T*Dh,R.data()+b*T*T,T,T,Dh);
            emu_bmm_nt_global(q.data(),k.data(),G.data(),BH,T,T,Dh,T*Dh,T*Dh,T*T,1.f);
            emu_bmm_nt_shared(q.data(),k.data(),S.data(),BH,T,T,Dh,T*Dh,T*Dh,T*T,1.f);
            char nm[96];
            snprintf(nm,sizeof nm,"QK^T  T=%2d  global vs ref",T); check(nm,maxdiff(R,G),1e-3);
            snprintf(nm,sizeof nm,"QK^T  T=%2d  shared vs ref",T); check(nm,maxdiff(R,S),1e-3);
        }
        // P*V : bmm_nn
        {
            auto P=rnd(BH*T*T),v=rnd(BH*T*Dh);
            std::vector<float> R(BH*T*Dh),G(BH*T*Dh),S(BH*T*Dh);
            for(int b=0;b<BH;++b) ref_nn(P.data()+b*T*T,v.data()+b*T*Dh,nullptr,R.data()+b*T*Dh,T,Dh,T);
            emu_bmm_nn_global(P.data(),v.data(),G.data(),BH,T,Dh,T,T*T,T*Dh,T*Dh,1.f);
            emu_bmm_nn_shared(P.data(),v.data(),S.data(),BH,T,Dh,T,T*T,T*Dh,T*Dh,1.f);
            char nm[96];
            snprintf(nm,sizeof nm,"P*V   T=%2d  global vs ref",T); check(nm,maxdiff(R,G),1e-3);
            snprintf(nm,sizeof nm,"P*V   T=%2d  shared vs ref",T); check(nm,maxdiff(R,S),1e-3);
        }
        // dV = P^T*dO : bmm_tn
        {
            auto P=rnd(BH*T*T),dO=rnd(BH*T*Dh);
            std::vector<float> R(BH*T*Dh),G(BH*T*Dh),S(BH*T*Dh);
            for(int b=0;b<BH;++b) ref_tn(P.data()+b*T*T,dO.data()+b*T*Dh,R.data()+b*T*Dh,T,Dh,T);
            emu_bmm_tn_global(P.data(),dO.data(),G.data(),BH,T,Dh,T,T*T,T*Dh,T*Dh,1.f);
            emu_bmm_tn_shared(P.data(),dO.data(),S.data(),BH,T,Dh,T,T*T,T*Dh,T*Dh,1.f);
            char nm[96];
            snprintf(nm,sizeof nm,"P^T*dO T=%2d global vs ref",T); check(nm,maxdiff(R,G),1e-3);
            snprintf(nm,sizeof nm,"P^T*dO T=%2d shared vs ref",T); check(nm,maxdiff(R,S),1e-3);
        }
    }

    printf("\n=== 3. Reordenamientos (ida y vuelta) ===\n");
    {
        int B=3,T=17,H=4,Dh=16,D=H*Dh;
        auto qkv=rnd(B*T*3*D);
        std::vector<float>q(B*T*D),k(B*T*D),v(B*T*D),back(B*T*3*D);
        emu_split_qkv(qkv.data(),q.data(),k.data(),v.data(),B,T,H,Dh);
        emu_merge_qkv_grad(q.data(),k.data(),v.data(),back.data(),B,T,H,Dh);
        check("split_qkv -> merge_qkv_grad == id",maxdiff(qkv,back),0.0);

        auto attn=rnd(B*T*D);
        std::vector<float>y(B*T*D),back2(B*T*D);
        emu_merge_heads(attn.data(),y.data(),B,T,H,Dh);
        emu_split_heads(y.data(),back2.data(),B,T,H,Dh);
        check("merge_heads -> split_heads == id",maxdiff(attn,back2),0.0);
    }

    printf("\n=== 4. im2patch (cobertura exacta de la imagen) ===\n");
    for(int patch:{4,7,14}){
        int B=2,S=28,gs=S/patch,pd=patch*patch,np=gs*gs;
        std::vector<float> img(B*S*S);
        for(int i=0;i<B*S*S;++i) img[i]=(float)i;          // patron identificable
        std::vector<float> pt(B*np*pd);
        emu_im2patch(img.data(),pt.data(),B,S,patch);
        // cada pixel debe aparecer exactamente una vez por imagen
        std::vector<int> cnt(B*S*S,0);
        for(int i=0;i<B*np*pd;++i){int val=(int)pt[i]; cnt[val]++;}
        int bad=0;for(int c:cnt) if(c!=1) bad++;
        char nm[96];snprintf(nm,sizeof nm,"im2patch patch=%2d biyectivo",patch);
        check(nm,(double)bad,0.0);
    }

    printf("\n=== 5. Gradientes analiticos vs numericos ===\n");
    // --- LayerNorm ---
    {
        int R=4,D=16;float eps=1e-5f;
        auto x=rnd(R*D),g=rnd(D),b=rnd(D),dy=rnd(R*D);
        std::vector<float>y(R*D),mean(R),rstd(R),dx(R*D),dg(D),db(D);
        ln_fwd(x.data(),g.data(),b.data(),y.data(),mean.data(),rstd.data(),R,D,eps);
        ln_bwd(dy.data(),x.data(),g.data(),mean.data(),rstd.data(),dx.data(),dg.data(),db.data(),R,D);
        // L = sum(dy*y); dL/dx numerico
        auto loss=[&](std::vector<float>&xx,std::vector<float>&gg,std::vector<float>&bb){
            std::vector<float>yy(R*D),mm(R),rr(R);
            ln_fwd(xx.data(),gg.data(),bb.data(),yy.data(),mm.data(),rr.data(),R,D,eps);
            double s=0;for(int i=0;i<R*D;++i)s+=(double)dy[i]*yy[i];return s;};
        double worst=0;const float h=1e-3f;
        for(int i=0;i<R*D;++i){auto xp=x,xm=x;xp[i]+=h;xm[i]-=h;
            double num=(loss(xp,g,b)-loss(xm,g,b))/(2*h);
            worst=std::max(worst,std::fabs(num-dx[i])/std::max(1.0,std::fabs(num)));}
        check("LayerNorm  dx  (err. relativo)",worst,2e-2);
        worst=0;
        for(int j=0;j<D;++j){auto gp=g,gm=g;gp[j]+=h;gm[j]-=h;
            double num=(loss(x,gp,b)-loss(x,gm,b))/(2*h);
            worst=std::max(worst,std::fabs(num-dg[j])/std::max(1.0,std::fabs(num)));}
        check("LayerNorm  dgamma (err. relativo)",worst,2e-2);
        worst=0;
        for(int j=0;j<D;++j){auto bp=b,bm=b;bp[j]+=h;bm[j]-=h;
            double num=(loss(x,g,bp)-loss(x,g,bm))/(2*h);
            worst=std::max(worst,std::fabs(num-db[j])/std::max(1.0,std::fabs(num)));}
        check("LayerNorm  dbeta  (err. relativo)",worst,2e-2);
    }
    // --- Softmax (con el factor de escala de la atencion) ---
    {
        int R=5,C=17;float scale=0.25f;
        auto s=rnd(R*C),dP=rnd(R*C);
        std::vector<float>P(R*C),dS(R*C);
        // el forward real es P = softmax(scale * s)
        std::vector<float> ss(R*C); for(int i=0;i<R*C;++i)ss[i]=scale*s[i];
        sm_fwd(ss.data(),P.data(),R,C);
        sm_bwd(P.data(),dP.data(),dS.data(),R,C,scale);
        auto loss=[&](std::vector<float>&sv){
            std::vector<float>t(R*C),p(R*C);for(int i=0;i<R*C;++i)t[i]=scale*sv[i];
            sm_fwd(t.data(),p.data(),R,C);
            double a=0;for(int i=0;i<R*C;++i)a+=(double)dP[i]*p[i];return a;};
        double worst=0;const float h=1e-3f;
        for(int i=0;i<R*C;++i){auto sp=s,sm=s;sp[i]+=h;sm[i]-=h;
            double num=(loss(sp)-loss(sm))/(2*h);
            worst=std::max(worst,std::fabs(num-dS[i])/std::max(1.0,std::fabs(num)));}
        check("Softmax(scale*s)  dS (err. relativo)",worst,2e-2);
    }
    // --- GELU ---
    {
        double worst=0;const float h=1e-3f;
        for(float v=-3.f;v<=3.f;v+=0.13f){
            double num=(gelu(v+h)-gelu(v-h))/(2*h);
            worst=std::max(worst,std::fabs(num-gelu_d(v))/std::max(1.0,std::fabs(num)));}
        check("GELU  derivada (err. relativo)",worst,1e-3);
    }

    printf("\n==================================================\n");
    if(fails==0) printf("TODOS LOS CHEQUEOS PASARON\n");
    else         printf("%d CHEQUEO(S) FALLARON\n",fails);
    printf("==================================================\n\n");
    return fails?1:0;
}
