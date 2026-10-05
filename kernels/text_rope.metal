// Text-only specialization of mlx-vlm rope_utils.py (MIT).
uint elem=thread_position_in_grid.x;
const int half_dim=ROTARY/2;
const int bsz=x_shape[0], heads=x_shape[1], len=x_shape[2], dim=x_shape[3];
const int slots=half_dim+dim-ROTARY;
if(elem>=uint(bsz*heads*len*slots)) return;
int local=int(elem),slot=local%slots,tmp=local/slots,t=tmp%len;
tmp/=len;int h=tmp%heads,b=tmp/heads;
int base=((b*heads+h)*len+t)*dim;
if(slot>=half_dim){int d=ROTARY+slot-half_dim;x_out[base+d]=x[base+d];return;}
float pos=float(position_ids[b*len+t]);float angle=pos*float(inv_freq[slot]);
float c=metal::cos(angle),s=metal::sin(angle);
float xv=float(x[base+slot]),xp=float(x[base+slot+half_dim]);
x_out[base+slot]=T(xv*c-xp*s);x_out[base+slot+half_dim]=T(xp*c+xv*s);
