clc
%设置参数  Brusselator系统的几类差分格式研究.pdf   p14
a=1;
b=2;
c=0;


xrange=1;
trange=1;
N = 128;
x = linspace(0,xrange,N);
dt=0.00001;
t = linspace(0,trange,trange/dt+1);
dx=x(2)-x(1);
u=zeros(N,1);
v=zeros(N,1);
uu=[];
vv=[];

%赋初值
for j=1:N-1
   u(j,1)=initialu(x(j));
   v(j,1)=initialv(x(j));
end
%赋边值
u(1)=0;
u(end)=0;
v(1)=0;
v(end)=0;

for k=1:trange/dt-1
	if rem(k,1000)==0
        uu=[uu,u];
        vv=[vv,v];
    end
    utemp=u;
    for j=2:N-1
       u(j)=(c+u(j)^2*v(j)-(b+1)*u(j)...
           +a*(u(j+1)-2*u(j)+u(j-1))/dx^2+f(x(j),t(k)))*dt...
           +u(j);
       v(j)=(b*utemp(j)-utemp(j)^2*v(j)...
           +a*(v(j+1)-2*v(j)+v(j-1))/dx^2+g(x(j),t(k)))*dt...
           +v(j);
    end
end
figure(1)
uu=[uu;vv];
contourf(uu,'LineStyle','none');



save uu
function f=f(x,t)
    f=(pi^2+2)*exp(-t)*sin(pi*x)-x*(1-x)*exp(-3*t)*sin(pi*x)^2;
end
function g=g(x,t)
    g=x*(x-1)*exp(-t)-2*exp(-t)*sin(pi*x)+x*(1-x)*exp(-3*t)*sin(pi*x)+2*exp(-t);
end
function initialv=initialv(x)
    initialv=x*(1-x);
end
function initialu=initialu(x)
    initialu=sin(pi*x);
end
