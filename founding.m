function  sacral_slope = founding(S1,R_ASIS,L_ASIS,L_PSIS,R_PSIS)
%function to calculate angle
%normal to plane  that perpendicular to the upper S1 level passing through its centre
A=det([L_PSIS(2)-S1(2), L_PSIS(3)-S1(3) ;R_PSIS(2)-S1(2),R_PSIS(3)-S1(3)]);
B=det([L_PSIS(1)-S1(1), L_PSIS(3)-S1(3) ;R_PSIS(1)-S1(1),R_PSIS(3)-S1(3)]);
C=det([L_PSIS(1)-S1(1), L_PSIS(2)-S1(2) ;R_PSIS(1)-S1(1),R_PSIS(2)-S1(2)]);
%find an angle between two vectors in radians/change to degrees
norm_vector=(S1-R_ASIS-L_ASIS);
asind((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)))
acosd((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)))
norm_vector=-(S1-R_ASIS-L_ASIS);
asind((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)))
acosd((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)))
norm_vector=[0 0 1];
a1=asind((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));
a2=acosd((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));

norm_vector=[0 0 -1];
a3=asind((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));
a4=acosd((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));

norm_vector=[0 1 0];
b1=asind((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));
b2=acosd((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));
norm_vector=[0 -1 0];
b3=asind((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));
b4=acosd((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));

norm_vector=[ 1 0 0];
c1=asind((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));
c2=acosd((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));
norm_vector=[ -1 0 0];
c3=asind((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));
c4=acosd((norm_vector(1)*A+norm_vector(2)*B+norm_vector(3)*C)/(norm(norm_vector)*sqrt(A*A+B*B+C*C)));

a1
a2
a3
a4

b1
b2
b3
b4

c1
c2
c3
c4



a1+b1
a1+b2
a2+b1
a2+b2
a3+b1
a3+b2
a4+b1
a4+b2
a1+b3
a1+b4
a2+b3
a2+b4
a3+b3
a3+b4
a4+b3
a4+b4

c1+b1
c1+b2
c2+b1
c2+b2
c3+b1
c3+b2
c4+b1
c4+b2
c1+b3
c1+b4
c2+b3
c2+b4
c3+b3
c3+b4
c4+b3
c4+b4

a1+c1
a1+c2
a2+c1
a2+c2
a3+c1
a3+c2
a4+c1
a4+c2
a1+c3
a1+c4
a2+c3
a2+c4
a3+c3
a3+c4
a4+c3
a4+c4



a1-b1
a1-b2
a2-b1
a2-b2
a3-b1
a3-b2
a4-b1
a4-b2
a1-b3
a1-b4
a2-b3
a2-b4
a3-b3
a3-b4
a4-b3
a4-b4

c1-b1
c1-b2
c2-b1
c2-b2
c3-b1
c3-b2
c4-b1
c4-b2
c1-b3
c1-b4
c2-b3
c2-b4
c3-b3
c3-b4
c4-b3
c4-b4

a1-c1
a1-c2
a2-c1
a2-c2
a3-c1
a3-c2
a4-c1
a4-c2
a1-c3
a1-c4
a2-c3
a2-c4
a3-c3
a3-c4
a4-c3
a4-c4

sacral_slope=0
end