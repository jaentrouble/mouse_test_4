��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*	2.3.0-rc12v2.3.0-rc0-15-g14b2d686d68ޱ
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	�@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:	@*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:@*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:@ *
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
: *
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
: *
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:	@*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:@*
dtype0
~
conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv1d_4/kernel
w
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*"
_output_shapes
:@ *
dtype0
r
conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_4/bias
k
!conv1d_4/bias/Read/ReadVariableOpReadVariableOpconv1d_4/bias*
_output_shapes
: *
dtype0
~
conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1d_5/kernel
w
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*"
_output_shapes
: *
dtype0
r
conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_5/bias
k
!conv1d_5/bias/Read/ReadVariableOpReadVariableOpconv1d_5/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
��*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/dense_1/kernel/m
�
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
��*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*&
shared_nameAdam/dense_2/kernel/m
�
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	�@*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*%
shared_nameAdam/conv1d/kernel/m
�
(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
:	@*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv1d_1/kernel/m
�
*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*"
_output_shapes
:@ *
dtype0
�
Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_2/kernel/m
�
*Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/m*"
_output_shapes
: *
dtype0
�
Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_2/bias/m
y
(Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv1d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/conv1d_3/kernel/m
�
*Adam/conv1d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/m*"
_output_shapes
:	@*
dtype0
�
Adam/conv1d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d_3/bias/m
y
(Adam/conv1d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv1d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv1d_4/kernel/m
�
*Adam/conv1d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/kernel/m*"
_output_shapes
:@ *
dtype0
�
Adam/conv1d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_4/bias/m
y
(Adam/conv1d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv1d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_5/kernel/m
�
*Adam/conv1d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/kernel/m*"
_output_shapes
: *
dtype0
�
Adam/conv1d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_5/bias/m
y
(Adam/conv1d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
��*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/dense_1/kernel/v
�
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
��*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*&
shared_nameAdam/dense_2/kernel/v
�
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	�@*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*%
shared_nameAdam/conv1d/kernel/v
�
(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
:	@*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:@*
dtype0
�
Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv1d_1/kernel/v
�
*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*"
_output_shapes
:@ *
dtype0
�
Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_2/kernel/v
�
*Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/v*"
_output_shapes
: *
dtype0
�
Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_2/bias/v
y
(Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv1d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/conv1d_3/kernel/v
�
*Adam/conv1d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/v*"
_output_shapes
:	@*
dtype0
�
Adam/conv1d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d_3/bias/v
y
(Adam/conv1d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/v*
_output_shapes
:@*
dtype0
�
Adam/conv1d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv1d_4/kernel/v
�
*Adam/conv1d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/kernel/v*"
_output_shapes
:@ *
dtype0
�
Adam/conv1d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_4/bias/v
y
(Adam/conv1d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv1d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_5/kernel/v
�
*Adam/conv1d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/kernel/v*"
_output_shapes
: *
dtype0
�
Adam/conv1d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_5/bias/v
y
(Adam/conv1d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�v
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�v
value�vB�v B�v
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
#_self_saveable_object_factories
	optimizer
loss

signatures
	variables
regularization_losses
trainable_variables
	keras_api
%
#_self_saveable_object_factories
%
#_self_saveable_object_factories
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
�
 layer-0
!layer-1
"layer_with_weights-0
"layer-2
#layer_with_weights-1
#layer-3
$layer_with_weights-2
$layer-4
#%_self_saveable_object_factories
&	variables
'regularization_losses
(trainable_variables
)	keras_api
w
#*_self_saveable_object_factories
+	variables
,regularization_losses
-trainable_variables
.	keras_api
w
#/_self_saveable_object_factories
0	variables
1regularization_losses
2trainable_variables
3	keras_api
�

4kernel
5bias
#6_self_saveable_object_factories
7	variables
8regularization_losses
9trainable_variables
:	keras_api
�

;kernel
<bias
#=_self_saveable_object_factories
>	variables
?regularization_losses
@trainable_variables
A	keras_api
�

Bkernel
Cbias
#D_self_saveable_object_factories
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
�

Ikernel
Jbias
#K_self_saveable_object_factories
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
w
#P_self_saveable_object_factories
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
 
�
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_rate4m�5m�;m�<m�Bm�Cm�Im�Jm�Zm�[m�\m�]m�^m�_m�`m�am�bm�cm�dm�em�4v�5v�;v�<v�Bv�Cv�Iv�Jv�Zv�[v�\v�]v�^v�_v�`v�av�bv�cv�dv�ev�
 
 
�
Z0
[1
\2
]3
^4
_5
`6
a7
b8
c9
d10
e11
412
513
;14
<15
B16
C17
I18
J19
 
�
Z0
[1
\2
]3
^4
_5
`6
a7
b8
c9
d10
e11
412
513
;14
<15
B16
C17
I18
J19
�
	variables
regularization_losses
fnon_trainable_variables
glayer_metrics
hlayer_regularization_losses

ilayers
trainable_variables
jmetrics
 
 
%
#k_self_saveable_object_factories
w
#l_self_saveable_object_factories
m	variables
nregularization_losses
otrainable_variables
p	keras_api
�

Zkernel
[bias
#q_self_saveable_object_factories
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
�

\kernel
]bias
#v_self_saveable_object_factories
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
�

^kernel
_bias
#{_self_saveable_object_factories
|	variables
}regularization_losses
~trainable_variables
	keras_api
 
*
Z0
[1
\2
]3
^4
_5
 
*
Z0
[1
\2
]3
^4
_5
�
	variables
regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
trainable_variables
�metrics
&
$�_self_saveable_object_factories
|
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

`kernel
abias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

bkernel
cbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�

dkernel
ebias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
 
*
`0
a1
b2
c3
d4
e5
 
*
`0
a1
b2
c3
d4
e5
�
&	variables
'regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
(trainable_variables
�metrics
 
 
 
 
�
+	variables
,regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
-trainable_variables
�metrics
 
 
 
 
�
0	variables
1regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
2trainable_variables
�metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51
 

40
51
�
7	variables
8regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
9trainable_variables
�metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1
 

;0
<1
�
>	variables
?regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
@trainable_variables
�metrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1
 

B0
C1
�
E	variables
Fregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
Gtrainable_variables
�metrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1
 

I0
J1
�
L	variables
Mregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
Ntrainable_variables
�metrics
 
 
 
 
�
Q	variables
Rregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
Strainable_variables
�metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv1d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1d_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv1d_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv1d_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
N
0
1
2
3
4
5
6
7
	8

9
10

�0
 
 
 
 
 
�
m	variables
nregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
otrainable_variables
�metrics
 

Z0
[1
 

Z0
[1
�
r	variables
sregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
ttrainable_variables
�metrics
 

\0
]1
 

\0
]1
�
w	variables
xregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
ytrainable_variables
�metrics
 

^0
_1
 

^0
_1
�
|	variables
}regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
~trainable_variables
�metrics
 
 
 
#
0
1
2
3
4
 
 
 
 
 
 
�
�	variables
�regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
�trainable_variables
�metrics
 

`0
a1
 

`0
a1
�
�	variables
�regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
�trainable_variables
�metrics
 

b0
c1
 

b0
c1
�
�	variables
�regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
�trainable_variables
�metrics
 

d0
e1
 

d0
e1
�
�	variables
�regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
�trainable_variables
�metrics
 
 
 
#
 0
!1
"2
#3
$4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv1d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv1d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv1d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv1d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv1d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv1d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv1d_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv1d_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv1d_4/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv1d_4/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv1d_5/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv1d_5/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv1d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv1d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv1d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv1d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv1d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv1d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv1d_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv1d_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv1d_4/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv1d_4/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv1d_5/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv1d_5/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_LeftPlaceholder*/
_output_shapes
:���������d*
dtype0*$
shape:���������d
�
serving_default_RightPlaceholder*/
_output_shapes
:���������d*
dtype0*$
shape:���������d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_Leftserving_default_Rightconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� */
f*R(
&__inference_signature_wrapper_22708399
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp#conv1d_4/kernel/Read/ReadVariableOp!conv1d_4/bias/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp!conv1d_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp*Adam/conv1d_2/kernel/m/Read/ReadVariableOp(Adam/conv1d_2/bias/m/Read/ReadVariableOp*Adam/conv1d_3/kernel/m/Read/ReadVariableOp(Adam/conv1d_3/bias/m/Read/ReadVariableOp*Adam/conv1d_4/kernel/m/Read/ReadVariableOp(Adam/conv1d_4/bias/m/Read/ReadVariableOp*Adam/conv1d_5/kernel/m/Read/ReadVariableOp(Adam/conv1d_5/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp*Adam/conv1d_2/kernel/v/Read/ReadVariableOp(Adam/conv1d_2/bias/v/Read/ReadVariableOp*Adam/conv1d_3/kernel/v/Read/ReadVariableOp(Adam/conv1d_3/bias/v/Read/ReadVariableOp*Adam/conv1d_4/kernel/v/Read/ReadVariableOp(Adam/conv1d_4/bias/v/Read/ReadVariableOp*Adam/conv1d_5/kernel/v/Read/ReadVariableOp(Adam/conv1d_5/bias/v/Read/ReadVariableOpConst*P
TinI
G2E	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__traced_save_22709530
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biastotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/conv1d_2/kernel/mAdam/conv1d_2/bias/mAdam/conv1d_3/kernel/mAdam/conv1d_3/bias/mAdam/conv1d_4/kernel/mAdam/conv1d_4/bias/mAdam/conv1d_5/kernel/mAdam/conv1d_5/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/conv1d_2/kernel/vAdam/conv1d_2/bias/vAdam/conv1d_3/kernel/vAdam/conv1d_3/bias/vAdam/conv1d_4/kernel/vAdam/conv1d_4/bias/vAdam/conv1d_5/kernel/vAdam/conv1d_5/bias/v*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference__traced_restore_22709741��
�
�
F__inference_Left_eye_layer_call_and_return_conditional_losses_22707607

inputs
conv1d_22707591
conv1d_22707593
conv1d_1_22707596
conv1d_1_22707598
conv1d_2_22707601
conv1d_2_22707603
identity��conv1d/StatefulPartitionedCall� conv1d_1/StatefulPartitionedCall� conv1d_2/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_227074222
reshape/PartitionedCall�
conv1d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_22707591conv1d_22707593*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_227074462 
conv1d/StatefulPartitionedCall�
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_22707596conv1d_1_22707598*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_227074782"
 conv1d_1/StatefulPartitionedCall�
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_22707601conv1d_2_22707603*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_227075102"
 conv1d_2/StatefulPartitionedCall�
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
+__inference_Left_eye_layer_call_fn_22707622
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_227076072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������d
!
_user_specified_name	input_1
�.
�
J__inference_functional_1_layer_call_and_return_conditional_losses_22708300

inputs
inputs_1
left_eye_22708250
left_eye_22708252
left_eye_22708254
left_eye_22708256
left_eye_22708258
left_eye_22708260
right_eye_22708263
right_eye_22708265
right_eye_22708267
right_eye_22708269
right_eye_22708271
right_eye_22708273
dense_22708278
dense_22708280
dense_1_22708283
dense_1_22708285
dense_2_22708288
dense_2_22708290
dense_3_22708293
dense_3_22708295
identity�� Left_eye/StatefulPartitionedCall�!Right_eye/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
 Left_eye/StatefulPartitionedCallStatefulPartitionedCallinputsleft_eye_22708250left_eye_22708252left_eye_22708254left_eye_22708256left_eye_22708258left_eye_22708260*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_227076072"
 Left_eye/StatefulPartitionedCall�
!Right_eye/StatefulPartitionedCallStatefulPartitionedCallinputs_1right_eye_22708263right_eye_22708265right_eye_22708267right_eye_22708269right_eye_22708271right_eye_22708273*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_227078242#
!Right_eye/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall)Left_eye/StatefulPartitionedCall:output:0*Right_eye/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_227079452
concatenate/PartitionedCall�
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_227079602
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_22708278dense_22708280*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_227079792
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_22708283dense_1_22708285*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_227080062!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_22708288dense_2_22708290*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_227080332!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_22708293dense_3_22708295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_227080592!
dense_3/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_227080792
activation/PartitionedCall�
IdentityIdentity#activation/PartitionedCall:output:0!^Left_eye/StatefulPartitionedCall"^Right_eye/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d::::::::::::::::::::2D
 Left_eye/StatefulPartitionedCall Left_eye/StatefulPartitionedCall2F
!Right_eye/StatefulPartitionedCall!Right_eye/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
G__inference_Right_eye_layer_call_and_return_conditional_losses_22707744
input_2
conv1d_3_22707674
conv1d_3_22707676
conv1d_4_22707706
conv1d_4_22707708
conv1d_5_22707738
conv1d_5_22707740
identity�� conv1d_3/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_227076392
reshape_1/PartitionedCall�
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_3_22707674conv1d_3_22707676*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_227076632"
 conv1d_3/StatefulPartitionedCall�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_22707706conv1d_4_22707708*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_227076952"
 conv1d_4/StatefulPartitionedCall�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0conv1d_5_22707738conv1d_5_22707740*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_227077272"
 conv1d_5/StatefulPartitionedCall�
IdentityIdentity)conv1d_5/StatefulPartitionedCall:output:0!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������d
!
_user_specified_name	input_2
�
�
C__inference_dense_layer_call_and_return_conditional_losses_22707979

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_functional_1_layer_call_fn_22708243
left	
right
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleftrightunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_227082002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
/
_output_shapes
:���������d

_user_specified_nameLeft:VR
/
_output_shapes
:���������d

_user_specified_nameRight
�
�
F__inference_conv1d_3_layer_call_and_return_conditional_losses_22709246

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������d	:::S O
+
_output_shapes
:���������d	
 
_user_specified_nameinputs
�.
�
J__inference_functional_1_layer_call_and_return_conditional_losses_22708200

inputs
inputs_1
left_eye_22708150
left_eye_22708152
left_eye_22708154
left_eye_22708156
left_eye_22708158
left_eye_22708160
right_eye_22708163
right_eye_22708165
right_eye_22708167
right_eye_22708169
right_eye_22708171
right_eye_22708173
dense_22708178
dense_22708180
dense_1_22708183
dense_1_22708185
dense_2_22708188
dense_2_22708190
dense_3_22708193
dense_3_22708195
identity�� Left_eye/StatefulPartitionedCall�!Right_eye/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
 Left_eye/StatefulPartitionedCallStatefulPartitionedCallinputsleft_eye_22708150left_eye_22708152left_eye_22708154left_eye_22708156left_eye_22708158left_eye_22708160*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_227075702"
 Left_eye/StatefulPartitionedCall�
!Right_eye/StatefulPartitionedCallStatefulPartitionedCallinputs_1right_eye_22708163right_eye_22708165right_eye_22708167right_eye_22708169right_eye_22708171right_eye_22708173*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_227077872#
!Right_eye/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall)Left_eye/StatefulPartitionedCall:output:0*Right_eye/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_227079452
concatenate/PartitionedCall�
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_227079602
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_22708178dense_22708180*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_227079792
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_22708183dense_1_22708185*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_227080062!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_22708188dense_2_22708190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_227080332!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_22708193dense_3_22708195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_227080592!
dense_3/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_227080792
activation/PartitionedCall�
IdentityIdentity#activation/PartitionedCall:output:0!^Left_eye/StatefulPartitionedCall"^Right_eye/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d::::::::::::::::::::2D
 Left_eye/StatefulPartitionedCall Left_eye/StatefulPartitionedCall2F
!Right_eye/StatefulPartitionedCall!Right_eye/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
/__inference_functional_1_layer_call_fn_22708343
left	
right
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleftrightunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_227083002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
/
_output_shapes
:���������d

_user_specified_nameLeft:VR
/
_output_shapes
:���������d

_user_specified_nameRight
�
�
E__inference_dense_1_layer_call_and_return_conditional_losses_22708006

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv1d_4_layer_call_and_return_conditional_losses_22707695

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������/@:::S O
+
_output_shapes
:���������/@
 
_user_specified_nameinputs
�
F
*__inference_reshape_layer_call_fn_22709137

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_227074222
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������d	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
d
H__inference_activation_layer_call_and_return_conditional_losses_22708079

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
,__inference_reshape_1_layer_call_fn_22709230

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_227076392
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������d	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
F
*__inference_flatten_layer_call_fn_22709031

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_227079602
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������
 :S O
+
_output_shapes
:���������
 
 
_user_specified_nameinputs
�
�
D__inference_conv1d_layer_call_and_return_conditional_losses_22707446

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������d	:::S O
+
_output_shapes
:���������d	
 
_user_specified_nameinputs
�
�
+__inference_conv1d_1_layer_call_fn_22709187

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_227074782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������/@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������/@
 
_user_specified_nameinputs
�
�
F__inference_conv1d_2_layer_call_and_return_conditional_losses_22707510

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������
2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :::S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�	
J__inference_functional_1_layer_call_and_return_conditional_losses_22708651
inputs_0
inputs_1?
;left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource3
/left_eye_conv1d_biasadd_readvariableop_resourceA
=left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resource5
1left_eye_conv1d_1_biasadd_readvariableop_resourceA
=left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resource5
1left_eye_conv1d_2_biasadd_readvariableop_resourceB
>right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_3_biasadd_readvariableop_resourceB
>right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_4_biasadd_readvariableop_resourceB
>right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_5_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity�h
Left_eye/reshape/ShapeShapeinputs_0*
T0*
_output_shapes
:2
Left_eye/reshape/Shape�
$Left_eye/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Left_eye/reshape/strided_slice/stack�
&Left_eye/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Left_eye/reshape/strided_slice/stack_1�
&Left_eye/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Left_eye/reshape/strided_slice/stack_2�
Left_eye/reshape/strided_sliceStridedSliceLeft_eye/reshape/Shape:output:0-Left_eye/reshape/strided_slice/stack:output:0/Left_eye/reshape/strided_slice/stack_1:output:0/Left_eye/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Left_eye/reshape/strided_slice�
 Left_eye/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2"
 Left_eye/reshape/Reshape/shape/1�
 Left_eye/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2"
 Left_eye/reshape/Reshape/shape/2�
Left_eye/reshape/Reshape/shapePack'Left_eye/reshape/strided_slice:output:0)Left_eye/reshape/Reshape/shape/1:output:0)Left_eye/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2 
Left_eye/reshape/Reshape/shape�
Left_eye/reshape/ReshapeReshapeinputs_0'Left_eye/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:���������d	2
Left_eye/reshape/Reshape�
%Left_eye/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%Left_eye/conv1d/conv1d/ExpandDims/dim�
!Left_eye/conv1d/conv1d/ExpandDims
ExpandDims!Left_eye/reshape/Reshape:output:0.Left_eye/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2#
!Left_eye/conv1d/conv1d/ExpandDims�
2Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype024
2Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOp�
'Left_eye/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'Left_eye/conv1d/conv1d/ExpandDims_1/dim�
#Left_eye/conv1d/conv1d/ExpandDims_1
ExpandDims:Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:00Left_eye/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2%
#Left_eye/conv1d/conv1d/ExpandDims_1�
Left_eye/conv1d/conv1dConv2D*Left_eye/conv1d/conv1d/ExpandDims:output:0,Left_eye/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
Left_eye/conv1d/conv1d�
Left_eye/conv1d/conv1d/SqueezeSqueezeLeft_eye/conv1d/conv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2 
Left_eye/conv1d/conv1d/Squeeze�
&Left_eye/conv1d/BiasAdd/ReadVariableOpReadVariableOp/left_eye_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&Left_eye/conv1d/BiasAdd/ReadVariableOp�
Left_eye/conv1d/BiasAddBiasAdd'Left_eye/conv1d/conv1d/Squeeze:output:0.Left_eye/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2
Left_eye/conv1d/BiasAdd�
Left_eye/conv1d/ReluRelu Left_eye/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
Left_eye/conv1d/Relu�
'Left_eye/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'Left_eye/conv1d_1/conv1d/ExpandDims/dim�
#Left_eye/conv1d_1/conv1d/ExpandDims
ExpandDims"Left_eye/conv1d/Relu:activations:00Left_eye/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2%
#Left_eye/conv1d_1/conv1d/ExpandDims�
4Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype026
4Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp�
)Left_eye/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)Left_eye/conv1d_1/conv1d/ExpandDims_1/dim�
%Left_eye/conv1d_1/conv1d/ExpandDims_1
ExpandDims<Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:02Left_eye/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2'
%Left_eye/conv1d_1/conv1d/ExpandDims_1�
Left_eye/conv1d_1/conv1dConv2D,Left_eye/conv1d_1/conv1d/ExpandDims:output:0.Left_eye/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Left_eye/conv1d_1/conv1d�
 Left_eye/conv1d_1/conv1d/SqueezeSqueeze!Left_eye/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2"
 Left_eye/conv1d_1/conv1d/Squeeze�
(Left_eye/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp1left_eye_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Left_eye/conv1d_1/BiasAdd/ReadVariableOp�
Left_eye/conv1d_1/BiasAddBiasAdd)Left_eye/conv1d_1/conv1d/Squeeze:output:00Left_eye/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
Left_eye/conv1d_1/BiasAdd�
Left_eye/conv1d_1/ReluRelu"Left_eye/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Left_eye/conv1d_1/Relu�
'Left_eye/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'Left_eye/conv1d_2/conv1d/ExpandDims/dim�
#Left_eye/conv1d_2/conv1d/ExpandDims
ExpandDims$Left_eye/conv1d_1/Relu:activations:00Left_eye/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2%
#Left_eye/conv1d_2/conv1d/ExpandDims�
4Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype026
4Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp�
)Left_eye/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)Left_eye/conv1d_2/conv1d/ExpandDims_1/dim�
%Left_eye/conv1d_2/conv1d/ExpandDims_1
ExpandDims<Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:02Left_eye/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2'
%Left_eye/conv1d_2/conv1d/ExpandDims_1�
Left_eye/conv1d_2/conv1dConv2D,Left_eye/conv1d_2/conv1d/ExpandDims:output:0.Left_eye/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
Left_eye/conv1d_2/conv1d�
 Left_eye/conv1d_2/conv1d/SqueezeSqueeze!Left_eye/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2"
 Left_eye/conv1d_2/conv1d/Squeeze�
(Left_eye/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp1left_eye_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Left_eye/conv1d_2/BiasAdd/ReadVariableOp�
Left_eye/conv1d_2/BiasAddBiasAdd)Left_eye/conv1d_2/conv1d/Squeeze:output:00Left_eye/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2
Left_eye/conv1d_2/BiasAdd�
Left_eye/conv1d_2/ReluRelu"Left_eye/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������
2
Left_eye/conv1d_2/Relun
Right_eye/reshape_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2
Right_eye/reshape_1/Shape�
'Right_eye/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Right_eye/reshape_1/strided_slice/stack�
)Right_eye/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Right_eye/reshape_1/strided_slice/stack_1�
)Right_eye/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Right_eye/reshape_1/strided_slice/stack_2�
!Right_eye/reshape_1/strided_sliceStridedSlice"Right_eye/reshape_1/Shape:output:00Right_eye/reshape_1/strided_slice/stack:output:02Right_eye/reshape_1/strided_slice/stack_1:output:02Right_eye/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Right_eye/reshape_1/strided_slice�
#Right_eye/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2%
#Right_eye/reshape_1/Reshape/shape/1�
#Right_eye/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2%
#Right_eye/reshape_1/Reshape/shape/2�
!Right_eye/reshape_1/Reshape/shapePack*Right_eye/reshape_1/strided_slice:output:0,Right_eye/reshape_1/Reshape/shape/1:output:0,Right_eye/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!Right_eye/reshape_1/Reshape/shape�
Right_eye/reshape_1/ReshapeReshapeinputs_1*Right_eye/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:���������d	2
Right_eye/reshape_1/Reshape�
(Right_eye/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(Right_eye/conv1d_3/conv1d/ExpandDims/dim�
$Right_eye/conv1d_3/conv1d/ExpandDims
ExpandDims$Right_eye/reshape_1/Reshape:output:01Right_eye/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2&
$Right_eye/conv1d_3/conv1d/ExpandDims�
5Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype027
5Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp�
*Right_eye/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_3/conv1d/ExpandDims_1/dim�
&Right_eye/conv1d_3/conv1d/ExpandDims_1
ExpandDims=Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:03Right_eye/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2(
&Right_eye/conv1d_3/conv1d/ExpandDims_1�
Right_eye/conv1d_3/conv1dConv2D-Right_eye/conv1d_3/conv1d/ExpandDims:output:0/Right_eye/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
Right_eye/conv1d_3/conv1d�
!Right_eye/conv1d_3/conv1d/SqueezeSqueeze"Right_eye/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2#
!Right_eye/conv1d_3/conv1d/Squeeze�
)Right_eye/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)Right_eye/conv1d_3/BiasAdd/ReadVariableOp�
Right_eye/conv1d_3/BiasAddBiasAdd*Right_eye/conv1d_3/conv1d/Squeeze:output:01Right_eye/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2
Right_eye/conv1d_3/BiasAdd�
Right_eye/conv1d_3/ReluRelu#Right_eye/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
Right_eye/conv1d_3/Relu�
(Right_eye/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(Right_eye/conv1d_4/conv1d/ExpandDims/dim�
$Right_eye/conv1d_4/conv1d/ExpandDims
ExpandDims%Right_eye/conv1d_3/Relu:activations:01Right_eye/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2&
$Right_eye/conv1d_4/conv1d/ExpandDims�
5Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype027
5Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�
*Right_eye/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_4/conv1d/ExpandDims_1/dim�
&Right_eye/conv1d_4/conv1d/ExpandDims_1
ExpandDims=Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:03Right_eye/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2(
&Right_eye/conv1d_4/conv1d/ExpandDims_1�
Right_eye/conv1d_4/conv1dConv2D-Right_eye/conv1d_4/conv1d/ExpandDims:output:0/Right_eye/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Right_eye/conv1d_4/conv1d�
!Right_eye/conv1d_4/conv1d/SqueezeSqueeze"Right_eye/conv1d_4/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2#
!Right_eye/conv1d_4/conv1d/Squeeze�
)Right_eye/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)Right_eye/conv1d_4/BiasAdd/ReadVariableOp�
Right_eye/conv1d_4/BiasAddBiasAdd*Right_eye/conv1d_4/conv1d/Squeeze:output:01Right_eye/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
Right_eye/conv1d_4/BiasAdd�
Right_eye/conv1d_4/ReluRelu#Right_eye/conv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Right_eye/conv1d_4/Relu�
(Right_eye/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(Right_eye/conv1d_5/conv1d/ExpandDims/dim�
$Right_eye/conv1d_5/conv1d/ExpandDims
ExpandDims%Right_eye/conv1d_4/Relu:activations:01Right_eye/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2&
$Right_eye/conv1d_5/conv1d/ExpandDims�
5Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�
*Right_eye/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_5/conv1d/ExpandDims_1/dim�
&Right_eye/conv1d_5/conv1d/ExpandDims_1
ExpandDims=Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:03Right_eye/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2(
&Right_eye/conv1d_5/conv1d/ExpandDims_1�
Right_eye/conv1d_5/conv1dConv2D-Right_eye/conv1d_5/conv1d/ExpandDims:output:0/Right_eye/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
Right_eye/conv1d_5/conv1d�
!Right_eye/conv1d_5/conv1d/SqueezeSqueeze"Right_eye/conv1d_5/conv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2#
!Right_eye/conv1d_5/conv1d/Squeeze�
)Right_eye/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Right_eye/conv1d_5/BiasAdd/ReadVariableOp�
Right_eye/conv1d_5/BiasAddBiasAdd*Right_eye/conv1d_5/conv1d/Squeeze:output:01Right_eye/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2
Right_eye/conv1d_5/BiasAdd�
Right_eye/conv1d_5/ReluRelu#Right_eye/conv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������
2
Right_eye/conv1d_5/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2$Left_eye/conv1d_2/Relu:activations:0%Right_eye/conv1d_5/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������
 2
concatenate/concato
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
flatten/Const�
flatten/ReshapeReshapeconcatenate/concat:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_1/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_2/Relu�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/BiasAddl
IdentityIdentitydense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d:::::::::::::::::::::Y U
/
_output_shapes
:���������d
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������d
"
_user_specified_name
inputs/1
�
�
F__inference_conv1d_1_layer_call_and_return_conditional_losses_22707478

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������/@:::S O
+
_output_shapes
:���������/@
 
_user_specified_nameinputs
�
�
E__inference_dense_3_layer_call_and_return_conditional_losses_22708059

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
s
I__inference_concatenate_layer_call_and_return_conditional_losses_22707945

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:���������
 2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:���������
 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:���������
:���������
:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs:SO
+
_output_shapes
:���������

 
_user_specified_nameinputs
��
�	
J__inference_functional_1_layer_call_and_return_conditional_losses_22708525
inputs_0
inputs_1?
;left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource3
/left_eye_conv1d_biasadd_readvariableop_resourceA
=left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resource5
1left_eye_conv1d_1_biasadd_readvariableop_resourceA
=left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resource5
1left_eye_conv1d_2_biasadd_readvariableop_resourceB
>right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_3_biasadd_readvariableop_resourceB
>right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_4_biasadd_readvariableop_resourceB
>right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_5_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity�h
Left_eye/reshape/ShapeShapeinputs_0*
T0*
_output_shapes
:2
Left_eye/reshape/Shape�
$Left_eye/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Left_eye/reshape/strided_slice/stack�
&Left_eye/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Left_eye/reshape/strided_slice/stack_1�
&Left_eye/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Left_eye/reshape/strided_slice/stack_2�
Left_eye/reshape/strided_sliceStridedSliceLeft_eye/reshape/Shape:output:0-Left_eye/reshape/strided_slice/stack:output:0/Left_eye/reshape/strided_slice/stack_1:output:0/Left_eye/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Left_eye/reshape/strided_slice�
 Left_eye/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2"
 Left_eye/reshape/Reshape/shape/1�
 Left_eye/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2"
 Left_eye/reshape/Reshape/shape/2�
Left_eye/reshape/Reshape/shapePack'Left_eye/reshape/strided_slice:output:0)Left_eye/reshape/Reshape/shape/1:output:0)Left_eye/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2 
Left_eye/reshape/Reshape/shape�
Left_eye/reshape/ReshapeReshapeinputs_0'Left_eye/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:���������d	2
Left_eye/reshape/Reshape�
%Left_eye/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%Left_eye/conv1d/conv1d/ExpandDims/dim�
!Left_eye/conv1d/conv1d/ExpandDims
ExpandDims!Left_eye/reshape/Reshape:output:0.Left_eye/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2#
!Left_eye/conv1d/conv1d/ExpandDims�
2Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype024
2Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOp�
'Left_eye/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'Left_eye/conv1d/conv1d/ExpandDims_1/dim�
#Left_eye/conv1d/conv1d/ExpandDims_1
ExpandDims:Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:00Left_eye/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2%
#Left_eye/conv1d/conv1d/ExpandDims_1�
Left_eye/conv1d/conv1dConv2D*Left_eye/conv1d/conv1d/ExpandDims:output:0,Left_eye/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
Left_eye/conv1d/conv1d�
Left_eye/conv1d/conv1d/SqueezeSqueezeLeft_eye/conv1d/conv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2 
Left_eye/conv1d/conv1d/Squeeze�
&Left_eye/conv1d/BiasAdd/ReadVariableOpReadVariableOp/left_eye_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&Left_eye/conv1d/BiasAdd/ReadVariableOp�
Left_eye/conv1d/BiasAddBiasAdd'Left_eye/conv1d/conv1d/Squeeze:output:0.Left_eye/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2
Left_eye/conv1d/BiasAdd�
Left_eye/conv1d/ReluRelu Left_eye/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
Left_eye/conv1d/Relu�
'Left_eye/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'Left_eye/conv1d_1/conv1d/ExpandDims/dim�
#Left_eye/conv1d_1/conv1d/ExpandDims
ExpandDims"Left_eye/conv1d/Relu:activations:00Left_eye/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2%
#Left_eye/conv1d_1/conv1d/ExpandDims�
4Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype026
4Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp�
)Left_eye/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)Left_eye/conv1d_1/conv1d/ExpandDims_1/dim�
%Left_eye/conv1d_1/conv1d/ExpandDims_1
ExpandDims<Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:02Left_eye/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2'
%Left_eye/conv1d_1/conv1d/ExpandDims_1�
Left_eye/conv1d_1/conv1dConv2D,Left_eye/conv1d_1/conv1d/ExpandDims:output:0.Left_eye/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Left_eye/conv1d_1/conv1d�
 Left_eye/conv1d_1/conv1d/SqueezeSqueeze!Left_eye/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2"
 Left_eye/conv1d_1/conv1d/Squeeze�
(Left_eye/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp1left_eye_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Left_eye/conv1d_1/BiasAdd/ReadVariableOp�
Left_eye/conv1d_1/BiasAddBiasAdd)Left_eye/conv1d_1/conv1d/Squeeze:output:00Left_eye/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
Left_eye/conv1d_1/BiasAdd�
Left_eye/conv1d_1/ReluRelu"Left_eye/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Left_eye/conv1d_1/Relu�
'Left_eye/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'Left_eye/conv1d_2/conv1d/ExpandDims/dim�
#Left_eye/conv1d_2/conv1d/ExpandDims
ExpandDims$Left_eye/conv1d_1/Relu:activations:00Left_eye/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2%
#Left_eye/conv1d_2/conv1d/ExpandDims�
4Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype026
4Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp�
)Left_eye/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)Left_eye/conv1d_2/conv1d/ExpandDims_1/dim�
%Left_eye/conv1d_2/conv1d/ExpandDims_1
ExpandDims<Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:02Left_eye/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2'
%Left_eye/conv1d_2/conv1d/ExpandDims_1�
Left_eye/conv1d_2/conv1dConv2D,Left_eye/conv1d_2/conv1d/ExpandDims:output:0.Left_eye/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
Left_eye/conv1d_2/conv1d�
 Left_eye/conv1d_2/conv1d/SqueezeSqueeze!Left_eye/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2"
 Left_eye/conv1d_2/conv1d/Squeeze�
(Left_eye/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp1left_eye_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Left_eye/conv1d_2/BiasAdd/ReadVariableOp�
Left_eye/conv1d_2/BiasAddBiasAdd)Left_eye/conv1d_2/conv1d/Squeeze:output:00Left_eye/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2
Left_eye/conv1d_2/BiasAdd�
Left_eye/conv1d_2/ReluRelu"Left_eye/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������
2
Left_eye/conv1d_2/Relun
Right_eye/reshape_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2
Right_eye/reshape_1/Shape�
'Right_eye/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Right_eye/reshape_1/strided_slice/stack�
)Right_eye/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Right_eye/reshape_1/strided_slice/stack_1�
)Right_eye/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Right_eye/reshape_1/strided_slice/stack_2�
!Right_eye/reshape_1/strided_sliceStridedSlice"Right_eye/reshape_1/Shape:output:00Right_eye/reshape_1/strided_slice/stack:output:02Right_eye/reshape_1/strided_slice/stack_1:output:02Right_eye/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Right_eye/reshape_1/strided_slice�
#Right_eye/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2%
#Right_eye/reshape_1/Reshape/shape/1�
#Right_eye/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2%
#Right_eye/reshape_1/Reshape/shape/2�
!Right_eye/reshape_1/Reshape/shapePack*Right_eye/reshape_1/strided_slice:output:0,Right_eye/reshape_1/Reshape/shape/1:output:0,Right_eye/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!Right_eye/reshape_1/Reshape/shape�
Right_eye/reshape_1/ReshapeReshapeinputs_1*Right_eye/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:���������d	2
Right_eye/reshape_1/Reshape�
(Right_eye/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(Right_eye/conv1d_3/conv1d/ExpandDims/dim�
$Right_eye/conv1d_3/conv1d/ExpandDims
ExpandDims$Right_eye/reshape_1/Reshape:output:01Right_eye/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2&
$Right_eye/conv1d_3/conv1d/ExpandDims�
5Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype027
5Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp�
*Right_eye/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_3/conv1d/ExpandDims_1/dim�
&Right_eye/conv1d_3/conv1d/ExpandDims_1
ExpandDims=Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:03Right_eye/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2(
&Right_eye/conv1d_3/conv1d/ExpandDims_1�
Right_eye/conv1d_3/conv1dConv2D-Right_eye/conv1d_3/conv1d/ExpandDims:output:0/Right_eye/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
Right_eye/conv1d_3/conv1d�
!Right_eye/conv1d_3/conv1d/SqueezeSqueeze"Right_eye/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2#
!Right_eye/conv1d_3/conv1d/Squeeze�
)Right_eye/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)Right_eye/conv1d_3/BiasAdd/ReadVariableOp�
Right_eye/conv1d_3/BiasAddBiasAdd*Right_eye/conv1d_3/conv1d/Squeeze:output:01Right_eye/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2
Right_eye/conv1d_3/BiasAdd�
Right_eye/conv1d_3/ReluRelu#Right_eye/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
Right_eye/conv1d_3/Relu�
(Right_eye/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(Right_eye/conv1d_4/conv1d/ExpandDims/dim�
$Right_eye/conv1d_4/conv1d/ExpandDims
ExpandDims%Right_eye/conv1d_3/Relu:activations:01Right_eye/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2&
$Right_eye/conv1d_4/conv1d/ExpandDims�
5Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype027
5Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�
*Right_eye/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_4/conv1d/ExpandDims_1/dim�
&Right_eye/conv1d_4/conv1d/ExpandDims_1
ExpandDims=Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:03Right_eye/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2(
&Right_eye/conv1d_4/conv1d/ExpandDims_1�
Right_eye/conv1d_4/conv1dConv2D-Right_eye/conv1d_4/conv1d/ExpandDims:output:0/Right_eye/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Right_eye/conv1d_4/conv1d�
!Right_eye/conv1d_4/conv1d/SqueezeSqueeze"Right_eye/conv1d_4/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2#
!Right_eye/conv1d_4/conv1d/Squeeze�
)Right_eye/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)Right_eye/conv1d_4/BiasAdd/ReadVariableOp�
Right_eye/conv1d_4/BiasAddBiasAdd*Right_eye/conv1d_4/conv1d/Squeeze:output:01Right_eye/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
Right_eye/conv1d_4/BiasAdd�
Right_eye/conv1d_4/ReluRelu#Right_eye/conv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Right_eye/conv1d_4/Relu�
(Right_eye/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(Right_eye/conv1d_5/conv1d/ExpandDims/dim�
$Right_eye/conv1d_5/conv1d/ExpandDims
ExpandDims%Right_eye/conv1d_4/Relu:activations:01Right_eye/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2&
$Right_eye/conv1d_5/conv1d/ExpandDims�
5Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�
*Right_eye/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_5/conv1d/ExpandDims_1/dim�
&Right_eye/conv1d_5/conv1d/ExpandDims_1
ExpandDims=Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:03Right_eye/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2(
&Right_eye/conv1d_5/conv1d/ExpandDims_1�
Right_eye/conv1d_5/conv1dConv2D-Right_eye/conv1d_5/conv1d/ExpandDims:output:0/Right_eye/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
Right_eye/conv1d_5/conv1d�
!Right_eye/conv1d_5/conv1d/SqueezeSqueeze"Right_eye/conv1d_5/conv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2#
!Right_eye/conv1d_5/conv1d/Squeeze�
)Right_eye/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Right_eye/conv1d_5/BiasAdd/ReadVariableOp�
Right_eye/conv1d_5/BiasAddBiasAdd*Right_eye/conv1d_5/conv1d/Squeeze:output:01Right_eye/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2
Right_eye/conv1d_5/BiasAdd�
Right_eye/conv1d_5/ReluRelu#Right_eye/conv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������
2
Right_eye/conv1d_5/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2$Left_eye/conv1d_2/Relu:activations:0%Right_eye/conv1d_5/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������
 2
concatenate/concato
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
flatten/Const�
flatten/ReshapeReshapeconcatenate/concat:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_1/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_2/Relu�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/BiasAddl
IdentityIdentitydense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d:::::::::::::::::::::Y U
/
_output_shapes
:���������d
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������d
"
_user_specified_name
inputs/1
�
�
F__inference_Left_eye_layer_call_and_return_conditional_losses_22707547
input_1
conv1d_22707531
conv1d_22707533
conv1d_1_22707536
conv1d_1_22707538
conv1d_2_22707541
conv1d_2_22707543
identity��conv1d/StatefulPartitionedCall� conv1d_1/StatefulPartitionedCall� conv1d_2/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_227074222
reshape/PartitionedCall�
conv1d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_22707531conv1d_22707533*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_227074462 
conv1d/StatefulPartitionedCall�
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_22707536conv1d_1_22707538*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_227074782"
 conv1d_1/StatefulPartitionedCall�
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_22707541conv1d_2_22707543*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_227075102"
 conv1d_2/StatefulPartitionedCall�
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:X T
/
_output_shapes
:���������d
!
_user_specified_name	input_1
�
Z
.__inference_concatenate_layer_call_fn_22709020
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_227079452
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������
 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:���������
:���������
:U Q
+
_output_shapes
:���������

"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������

"
_user_specified_name
inputs/1
�
�
C__inference_dense_layer_call_and_return_conditional_losses_22709042

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�:
�
G__inference_Right_eye_layer_call_and_return_conditional_losses_22708924

inputs8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource8
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource
identity�X
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_1/Shape�
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack�
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1�
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/2�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape�
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:���������d	2
reshape_1/Reshape�
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_3/conv1d/ExpandDims/dim�
conv1d_3/conv1d/ExpandDims
ExpandDimsreshape_1/Reshape:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2
conv1d_3/conv1d/ExpandDims�
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim�
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d_3/conv1d/ExpandDims_1�
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
conv1d_3/conv1d�
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2
conv1d_3/conv1d/Squeeze�
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp�
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
conv1d_3/Relu�
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_4/conv1d/ExpandDims/dim�
conv1d_4/conv1d/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2
conv1d_4/conv1d/ExpandDims�
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dim�
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_4/conv1d/ExpandDims_1�
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_4/conv1d�
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_4/conv1d/Squeeze�
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_4/BiasAdd/ReadVariableOp�
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_4/BiasAddw
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_4/Relu�
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_5/conv1d/ExpandDims/dim�
conv1d_5/conv1d/ExpandDims
ExpandDimsconv1d_4/Relu:activations:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d_5/conv1d/ExpandDims�
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dim�
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_5/conv1d/ExpandDims_1�
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
conv1d_5/conv1d�
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2
conv1d_5/conv1d/Squeeze�
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_5/BiasAdd/ReadVariableOp�
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2
conv1d_5/BiasAddw
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������
2
conv1d_5/Relus
IdentityIdentityconv1d_5/Relu:activations:0*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d:::::::W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
a
E__inference_flatten_layer_call_and_return_conditional_losses_22709026

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������
 :S O
+
_output_shapes
:���������
 
 
_user_specified_nameinputs
��
�
#__inference__wrapped_model_22707405
left	
rightL
Hfunctional_1_left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource@
<functional_1_left_eye_conv1d_biasadd_readvariableop_resourceN
Jfunctional_1_left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resourceB
>functional_1_left_eye_conv1d_1_biasadd_readvariableop_resourceN
Jfunctional_1_left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resourceB
>functional_1_left_eye_conv1d_2_biasadd_readvariableop_resourceO
Kfunctional_1_right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resourceC
?functional_1_right_eye_conv1d_3_biasadd_readvariableop_resourceO
Kfunctional_1_right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resourceC
?functional_1_right_eye_conv1d_4_biasadd_readvariableop_resourceO
Kfunctional_1_right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resourceC
?functional_1_right_eye_conv1d_5_biasadd_readvariableop_resource5
1functional_1_dense_matmul_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource7
3functional_1_dense_1_matmul_readvariableop_resource8
4functional_1_dense_1_biasadd_readvariableop_resource7
3functional_1_dense_2_matmul_readvariableop_resource8
4functional_1_dense_2_biasadd_readvariableop_resource7
3functional_1_dense_3_matmul_readvariableop_resource8
4functional_1_dense_3_biasadd_readvariableop_resource
identity�~
#functional_1/Left_eye/reshape/ShapeShapeleft*
T0*
_output_shapes
:2%
#functional_1/Left_eye/reshape/Shape�
1functional_1/Left_eye/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1functional_1/Left_eye/reshape/strided_slice/stack�
3functional_1/Left_eye/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_1/Left_eye/reshape/strided_slice/stack_1�
3functional_1/Left_eye/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_1/Left_eye/reshape/strided_slice/stack_2�
+functional_1/Left_eye/reshape/strided_sliceStridedSlice,functional_1/Left_eye/reshape/Shape:output:0:functional_1/Left_eye/reshape/strided_slice/stack:output:0<functional_1/Left_eye/reshape/strided_slice/stack_1:output:0<functional_1/Left_eye/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+functional_1/Left_eye/reshape/strided_slice�
-functional_1/Left_eye/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2/
-functional_1/Left_eye/reshape/Reshape/shape/1�
-functional_1/Left_eye/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2/
-functional_1/Left_eye/reshape/Reshape/shape/2�
+functional_1/Left_eye/reshape/Reshape/shapePack4functional_1/Left_eye/reshape/strided_slice:output:06functional_1/Left_eye/reshape/Reshape/shape/1:output:06functional_1/Left_eye/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+functional_1/Left_eye/reshape/Reshape/shape�
%functional_1/Left_eye/reshape/ReshapeReshapeleft4functional_1/Left_eye/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:���������d	2'
%functional_1/Left_eye/reshape/Reshape�
2functional_1/Left_eye/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2functional_1/Left_eye/conv1d/conv1d/ExpandDims/dim�
.functional_1/Left_eye/conv1d/conv1d/ExpandDims
ExpandDims.functional_1/Left_eye/reshape/Reshape:output:0;functional_1/Left_eye/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	20
.functional_1/Left_eye/conv1d/conv1d/ExpandDims�
?functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHfunctional_1_left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02A
?functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOp�
4functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/dim�
0functional_1/Left_eye/conv1d/conv1d/ExpandDims_1
ExpandDimsGfunctional_1/Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0=functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@22
0functional_1/Left_eye/conv1d/conv1d/ExpandDims_1�
#functional_1/Left_eye/conv1d/conv1dConv2D7functional_1/Left_eye/conv1d/conv1d/ExpandDims:output:09functional_1/Left_eye/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2%
#functional_1/Left_eye/conv1d/conv1d�
+functional_1/Left_eye/conv1d/conv1d/SqueezeSqueeze,functional_1/Left_eye/conv1d/conv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2-
+functional_1/Left_eye/conv1d/conv1d/Squeeze�
3functional_1/Left_eye/conv1d/BiasAdd/ReadVariableOpReadVariableOp<functional_1_left_eye_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3functional_1/Left_eye/conv1d/BiasAdd/ReadVariableOp�
$functional_1/Left_eye/conv1d/BiasAddBiasAdd4functional_1/Left_eye/conv1d/conv1d/Squeeze:output:0;functional_1/Left_eye/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2&
$functional_1/Left_eye/conv1d/BiasAdd�
!functional_1/Left_eye/conv1d/ReluRelu-functional_1/Left_eye/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:���������/@2#
!functional_1/Left_eye/conv1d/Relu�
4functional_1/Left_eye/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������26
4functional_1/Left_eye/conv1d_1/conv1d/ExpandDims/dim�
0functional_1/Left_eye/conv1d_1/conv1d/ExpandDims
ExpandDims/functional_1/Left_eye/conv1d/Relu:activations:0=functional_1/Left_eye/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@22
0functional_1/Left_eye/conv1d_1/conv1d/ExpandDims�
Afunctional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpJfunctional_1_left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02C
Afunctional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp�
6functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 28
6functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/dim�
2functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1
ExpandDimsIfunctional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0?functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 24
2functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1�
%functional_1/Left_eye/conv1d_1/conv1dConv2D9functional_1/Left_eye/conv1d_1/conv1d/ExpandDims:output:0;functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2'
%functional_1/Left_eye/conv1d_1/conv1d�
-functional_1/Left_eye/conv1d_1/conv1d/SqueezeSqueeze.functional_1/Left_eye/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2/
-functional_1/Left_eye/conv1d_1/conv1d/Squeeze�
5functional_1/Left_eye/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp>functional_1_left_eye_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5functional_1/Left_eye/conv1d_1/BiasAdd/ReadVariableOp�
&functional_1/Left_eye/conv1d_1/BiasAddBiasAdd6functional_1/Left_eye/conv1d_1/conv1d/Squeeze:output:0=functional_1/Left_eye/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2(
&functional_1/Left_eye/conv1d_1/BiasAdd�
#functional_1/Left_eye/conv1d_1/ReluRelu/functional_1/Left_eye/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2%
#functional_1/Left_eye/conv1d_1/Relu�
4functional_1/Left_eye/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������26
4functional_1/Left_eye/conv1d_2/conv1d/ExpandDims/dim�
0functional_1/Left_eye/conv1d_2/conv1d/ExpandDims
ExpandDims1functional_1/Left_eye/conv1d_1/Relu:activations:0=functional_1/Left_eye/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 22
0functional_1/Left_eye/conv1d_2/conv1d/ExpandDims�
Afunctional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpJfunctional_1_left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02C
Afunctional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp�
6functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 28
6functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/dim�
2functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1
ExpandDimsIfunctional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0?functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 24
2functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1�
%functional_1/Left_eye/conv1d_2/conv1dConv2D9functional_1/Left_eye/conv1d_2/conv1d/ExpandDims:output:0;functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2'
%functional_1/Left_eye/conv1d_2/conv1d�
-functional_1/Left_eye/conv1d_2/conv1d/SqueezeSqueeze.functional_1/Left_eye/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2/
-functional_1/Left_eye/conv1d_2/conv1d/Squeeze�
5functional_1/Left_eye/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp>functional_1_left_eye_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5functional_1/Left_eye/conv1d_2/BiasAdd/ReadVariableOp�
&functional_1/Left_eye/conv1d_2/BiasAddBiasAdd6functional_1/Left_eye/conv1d_2/conv1d/Squeeze:output:0=functional_1/Left_eye/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2(
&functional_1/Left_eye/conv1d_2/BiasAdd�
#functional_1/Left_eye/conv1d_2/ReluRelu/functional_1/Left_eye/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������
2%
#functional_1/Left_eye/conv1d_2/Relu�
&functional_1/Right_eye/reshape_1/ShapeShaperight*
T0*
_output_shapes
:2(
&functional_1/Right_eye/reshape_1/Shape�
4functional_1/Right_eye/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4functional_1/Right_eye/reshape_1/strided_slice/stack�
6functional_1/Right_eye/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6functional_1/Right_eye/reshape_1/strided_slice/stack_1�
6functional_1/Right_eye/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6functional_1/Right_eye/reshape_1/strided_slice/stack_2�
.functional_1/Right_eye/reshape_1/strided_sliceStridedSlice/functional_1/Right_eye/reshape_1/Shape:output:0=functional_1/Right_eye/reshape_1/strided_slice/stack:output:0?functional_1/Right_eye/reshape_1/strided_slice/stack_1:output:0?functional_1/Right_eye/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.functional_1/Right_eye/reshape_1/strided_slice�
0functional_1/Right_eye/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d22
0functional_1/Right_eye/reshape_1/Reshape/shape/1�
0functional_1/Right_eye/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	22
0functional_1/Right_eye/reshape_1/Reshape/shape/2�
.functional_1/Right_eye/reshape_1/Reshape/shapePack7functional_1/Right_eye/reshape_1/strided_slice:output:09functional_1/Right_eye/reshape_1/Reshape/shape/1:output:09functional_1/Right_eye/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:20
.functional_1/Right_eye/reshape_1/Reshape/shape�
(functional_1/Right_eye/reshape_1/ReshapeReshaperight7functional_1/Right_eye/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:���������d	2*
(functional_1/Right_eye/reshape_1/Reshape�
5functional_1/Right_eye/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5functional_1/Right_eye/conv1d_3/conv1d/ExpandDims/dim�
1functional_1/Right_eye/conv1d_3/conv1d/ExpandDims
ExpandDims1functional_1/Right_eye/reshape_1/Reshape:output:0>functional_1/Right_eye/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	23
1functional_1/Right_eye/conv1d_3/conv1d/ExpandDims�
Bfunctional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKfunctional_1_right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02D
Bfunctional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp�
7functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/dim�
3functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1
ExpandDimsJfunctional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0@functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@25
3functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1�
&functional_1/Right_eye/conv1d_3/conv1dConv2D:functional_1/Right_eye/conv1d_3/conv1d/ExpandDims:output:0<functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2(
&functional_1/Right_eye/conv1d_3/conv1d�
.functional_1/Right_eye/conv1d_3/conv1d/SqueezeSqueeze/functional_1/Right_eye/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������20
.functional_1/Right_eye/conv1d_3/conv1d/Squeeze�
6functional_1/Right_eye/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp?functional_1_right_eye_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6functional_1/Right_eye/conv1d_3/BiasAdd/ReadVariableOp�
'functional_1/Right_eye/conv1d_3/BiasAddBiasAdd7functional_1/Right_eye/conv1d_3/conv1d/Squeeze:output:0>functional_1/Right_eye/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2)
'functional_1/Right_eye/conv1d_3/BiasAdd�
$functional_1/Right_eye/conv1d_3/ReluRelu0functional_1/Right_eye/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������/@2&
$functional_1/Right_eye/conv1d_3/Relu�
5functional_1/Right_eye/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5functional_1/Right_eye/conv1d_4/conv1d/ExpandDims/dim�
1functional_1/Right_eye/conv1d_4/conv1d/ExpandDims
ExpandDims2functional_1/Right_eye/conv1d_3/Relu:activations:0>functional_1/Right_eye/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@23
1functional_1/Right_eye/conv1d_4/conv1d/ExpandDims�
Bfunctional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKfunctional_1_right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02D
Bfunctional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�
7functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/dim�
3functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1
ExpandDimsJfunctional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0@functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 25
3functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1�
&functional_1/Right_eye/conv1d_4/conv1dConv2D:functional_1/Right_eye/conv1d_4/conv1d/ExpandDims:output:0<functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2(
&functional_1/Right_eye/conv1d_4/conv1d�
.functional_1/Right_eye/conv1d_4/conv1d/SqueezeSqueeze/functional_1/Right_eye/conv1d_4/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������20
.functional_1/Right_eye/conv1d_4/conv1d/Squeeze�
6functional_1/Right_eye/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp?functional_1_right_eye_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6functional_1/Right_eye/conv1d_4/BiasAdd/ReadVariableOp�
'functional_1/Right_eye/conv1d_4/BiasAddBiasAdd7functional_1/Right_eye/conv1d_4/conv1d/Squeeze:output:0>functional_1/Right_eye/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2)
'functional_1/Right_eye/conv1d_4/BiasAdd�
$functional_1/Right_eye/conv1d_4/ReluRelu0functional_1/Right_eye/conv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2&
$functional_1/Right_eye/conv1d_4/Relu�
5functional_1/Right_eye/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5functional_1/Right_eye/conv1d_5/conv1d/ExpandDims/dim�
1functional_1/Right_eye/conv1d_5/conv1d/ExpandDims
ExpandDims2functional_1/Right_eye/conv1d_4/Relu:activations:0>functional_1/Right_eye/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 23
1functional_1/Right_eye/conv1d_5/conv1d/ExpandDims�
Bfunctional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKfunctional_1_right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02D
Bfunctional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�
7functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/dim�
3functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1
ExpandDimsJfunctional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0@functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 25
3functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1�
&functional_1/Right_eye/conv1d_5/conv1dConv2D:functional_1/Right_eye/conv1d_5/conv1d/ExpandDims:output:0<functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2(
&functional_1/Right_eye/conv1d_5/conv1d�
.functional_1/Right_eye/conv1d_5/conv1d/SqueezeSqueeze/functional_1/Right_eye/conv1d_5/conv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������20
.functional_1/Right_eye/conv1d_5/conv1d/Squeeze�
6functional_1/Right_eye/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp?functional_1_right_eye_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6functional_1/Right_eye/conv1d_5/BiasAdd/ReadVariableOp�
'functional_1/Right_eye/conv1d_5/BiasAddBiasAdd7functional_1/Right_eye/conv1d_5/conv1d/Squeeze:output:0>functional_1/Right_eye/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2)
'functional_1/Right_eye/conv1d_5/BiasAdd�
$functional_1/Right_eye/conv1d_5/ReluRelu0functional_1/Right_eye/conv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������
2&
$functional_1/Right_eye/conv1d_5/Relu�
$functional_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_1/concatenate/concat/axis�
functional_1/concatenate/concatConcatV21functional_1/Left_eye/conv1d_2/Relu:activations:02functional_1/Right_eye/conv1d_5/Relu:activations:0-functional_1/concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������
 2!
functional_1/concatenate/concat�
functional_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
functional_1/flatten/Const�
functional_1/flatten/ReshapeReshape(functional_1/concatenate/concat:output:0#functional_1/flatten/Const:output:0*
T0*(
_output_shapes
:����������2
functional_1/flatten/Reshape�
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp1functional_1_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02*
(functional_1/dense/MatMul/ReadVariableOp�
functional_1/dense/MatMulMatMul%functional_1/flatten/Reshape:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
functional_1/dense/MatMul�
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOp�
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
functional_1/dense/BiasAdd�
functional_1/dense/ReluRelu#functional_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
functional_1/dense/Relu�
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOp�
functional_1/dense_1/MatMulMatMul%functional_1/dense/Relu:activations:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
functional_1/dense_1/MatMul�
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOp�
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
functional_1/dense_1/BiasAdd�
functional_1/dense_1/ReluRelu%functional_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
functional_1/dense_1/Relu�
*functional_1/dense_2/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02,
*functional_1/dense_2/MatMul/ReadVariableOp�
functional_1/dense_2/MatMulMatMul'functional_1/dense_1/Relu:activations:02functional_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
functional_1/dense_2/MatMul�
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOp�
functional_1/dense_2/BiasAddBiasAdd%functional_1/dense_2/MatMul:product:03functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
functional_1/dense_2/BiasAdd�
functional_1/dense_2/ReluRelu%functional_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
functional_1/dense_2/Relu�
*functional_1/dense_3/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*functional_1/dense_3/MatMul/ReadVariableOp�
functional_1/dense_3/MatMulMatMul'functional_1/dense_2/Relu:activations:02functional_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
functional_1/dense_3/MatMul�
+functional_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_1/dense_3/BiasAdd/ReadVariableOp�
functional_1/dense_3/BiasAddBiasAdd%functional_1/dense_3/MatMul:product:03functional_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
functional_1/dense_3/BiasAddy
IdentityIdentity%functional_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d:::::::::::::::::::::U Q
/
_output_shapes
:���������d

_user_specified_nameLeft:VR
/
_output_shapes
:���������d

_user_specified_nameRight
�
�
,__inference_Right_eye_layer_call_fn_22709007

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_227078242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
E__inference_dense_3_layer_call_and_return_conditional_losses_22709101

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:::O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
D__inference_conv1d_layer_call_and_return_conditional_losses_22709153

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������d	:::S O
+
_output_shapes
:���������d	
 
_user_specified_nameinputs
�9
�
F__inference_Left_eye_layer_call_and_return_conditional_losses_22708841

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource
identity�T
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/2�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*+
_output_shapes
:���������d	2
reshape/Reshape�
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/conv1d/ExpandDims/dim�
conv1d/conv1d/ExpandDims
ExpandDimsreshape/Reshape:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2
conv1d/conv1d/ExpandDims�
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp�
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim�
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/conv1d/ExpandDims_1�
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
conv1d/conv1d�
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2
conv1d/conv1d/Squeeze�
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1d/BiasAdd/ReadVariableOp�
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
conv1d/Relu�
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_1/conv1d/ExpandDims/dim�
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2
conv1d_1/conv1d/ExpandDims�
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim�
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_1/conv1d/ExpandDims_1�
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_1/conv1d�
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_1/conv1d/Squeeze�
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp�
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_1/Relu�
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_2/conv1d/ExpandDims/dim�
conv1d_2/conv1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d_2/conv1d/ExpandDims�
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim�
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_2/conv1d/ExpandDims_1�
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
conv1d_2/conv1d�
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2
conv1d_2/conv1d/Squeeze�
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp�
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������
2
conv1d_2/Relus
IdentityIdentityconv1d_2/Relu:activations:0*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d:::::::W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
+__inference_conv1d_2_layer_call_fn_22709212

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_227075102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_conv1d_4_layer_call_and_return_conditional_losses_22709271

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������/@:::S O
+
_output_shapes
:���������/@
 
_user_specified_nameinputs
�
a
E__inference_reshape_layer_call_and_return_conditional_losses_22709132

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape/shape/2�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������d	2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������d	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
+__inference_conv1d_5_layer_call_fn_22709305

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_227077272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_Left_eye_layer_call_fn_22708858

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_227075702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
E__inference_dense_2_layer_call_and_return_conditional_losses_22708033

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

*__inference_dense_2_layer_call_fn_22709091

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_227080332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

*__inference_dense_3_layer_call_fn_22709110

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_227080592
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�"
$__inference__traced_restore_22709741
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate%
!assignvariableop_13_conv1d_kernel#
assignvariableop_14_conv1d_bias'
#assignvariableop_15_conv1d_1_kernel%
!assignvariableop_16_conv1d_1_bias'
#assignvariableop_17_conv1d_2_kernel%
!assignvariableop_18_conv1d_2_bias'
#assignvariableop_19_conv1d_3_kernel%
!assignvariableop_20_conv1d_3_bias'
#assignvariableop_21_conv1d_4_kernel%
!assignvariableop_22_conv1d_4_bias'
#assignvariableop_23_conv1d_5_kernel%
!assignvariableop_24_conv1d_5_bias
assignvariableop_25_total
assignvariableop_26_count+
'assignvariableop_27_adam_dense_kernel_m)
%assignvariableop_28_adam_dense_bias_m-
)assignvariableop_29_adam_dense_1_kernel_m+
'assignvariableop_30_adam_dense_1_bias_m-
)assignvariableop_31_adam_dense_2_kernel_m+
'assignvariableop_32_adam_dense_2_bias_m-
)assignvariableop_33_adam_dense_3_kernel_m+
'assignvariableop_34_adam_dense_3_bias_m,
(assignvariableop_35_adam_conv1d_kernel_m*
&assignvariableop_36_adam_conv1d_bias_m.
*assignvariableop_37_adam_conv1d_1_kernel_m,
(assignvariableop_38_adam_conv1d_1_bias_m.
*assignvariableop_39_adam_conv1d_2_kernel_m,
(assignvariableop_40_adam_conv1d_2_bias_m.
*assignvariableop_41_adam_conv1d_3_kernel_m,
(assignvariableop_42_adam_conv1d_3_bias_m.
*assignvariableop_43_adam_conv1d_4_kernel_m,
(assignvariableop_44_adam_conv1d_4_bias_m.
*assignvariableop_45_adam_conv1d_5_kernel_m,
(assignvariableop_46_adam_conv1d_5_bias_m+
'assignvariableop_47_adam_dense_kernel_v)
%assignvariableop_48_adam_dense_bias_v-
)assignvariableop_49_adam_dense_1_kernel_v+
'assignvariableop_50_adam_dense_1_bias_v-
)assignvariableop_51_adam_dense_2_kernel_v+
'assignvariableop_52_adam_dense_2_bias_v-
)assignvariableop_53_adam_dense_3_kernel_v+
'assignvariableop_54_adam_dense_3_bias_v,
(assignvariableop_55_adam_conv1d_kernel_v*
&assignvariableop_56_adam_conv1d_bias_v.
*assignvariableop_57_adam_conv1d_1_kernel_v,
(assignvariableop_58_adam_conv1d_1_bias_v.
*assignvariableop_59_adam_conv1d_2_kernel_v,
(assignvariableop_60_adam_conv1d_2_bias_v.
*assignvariableop_61_adam_conv1d_3_kernel_v,
(assignvariableop_62_adam_conv1d_3_bias_v.
*assignvariableop_63_adam_conv1d_4_kernel_v,
(assignvariableop_64_adam_conv1d_4_bias_v.
*assignvariableop_65_adam_conv1d_5_kernel_v,
(assignvariableop_66_adam_conv1d_5_bias_v
identity_68��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*�!
value�!B�!DB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*�
value�B�DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv1d_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_conv1d_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv1d_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp!assignvariableop_16_conv1d_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp#assignvariableop_17_conv1d_2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_conv1d_2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp#assignvariableop_19_conv1d_3_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp!assignvariableop_20_conv1d_3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp#assignvariableop_21_conv1d_4_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp!assignvariableop_22_conv1d_4_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp#assignvariableop_23_conv1d_5_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp!assignvariableop_24_conv1d_5_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv1d_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_conv1d_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv1d_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv1d_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv1d_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv1d_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv1d_3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv1d_3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv1d_4_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv1d_4_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv1d_5_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv1d_5_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_dense_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp%assignvariableop_48_adam_dense_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_3_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_3_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_conv1d_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp&assignvariableop_56_adam_conv1d_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv1d_1_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv1d_1_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv1d_2_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv1d_2_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv1d_3_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv1d_3_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv1d_4_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv1d_4_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_conv1d_5_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_conv1d_5_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_669
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_67�
Identity_68IdentityIdentity_67:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_68"#
identity_68Identity_68:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�9
�
F__inference_Left_eye_layer_call_and_return_conditional_losses_22708792

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource
identity�T
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/2�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*+
_output_shapes
:���������d	2
reshape/Reshape�
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/conv1d/ExpandDims/dim�
conv1d/conv1d/ExpandDims
ExpandDimsreshape/Reshape:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2
conv1d/conv1d/ExpandDims�
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp�
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim�
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/conv1d/ExpandDims_1�
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
conv1d/conv1d�
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2
conv1d/conv1d/Squeeze�
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1d/BiasAdd/ReadVariableOp�
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
conv1d/Relu�
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_1/conv1d/ExpandDims/dim�
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2
conv1d_1/conv1d/ExpandDims�
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim�
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_1/conv1d/ExpandDims_1�
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_1/conv1d�
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_1/conv1d/Squeeze�
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp�
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_1/Relu�
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_2/conv1d/ExpandDims/dim�
conv1d_2/conv1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d_2/conv1d/ExpandDims�
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim�
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_2/conv1d/ExpandDims_1�
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
conv1d_2/conv1d�
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2
conv1d_2/conv1d/Squeeze�
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp�
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������
2
conv1d_2/Relus
IdentityIdentityconv1d_2/Relu:activations:0*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d:::::::W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
c
G__inference_reshape_1_layer_call_and_return_conditional_losses_22707639

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape/shape/2�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������d	2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������d	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
F__inference_conv1d_5_layer_call_and_return_conditional_losses_22707727

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������
2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :::S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_functional_1_layer_call_fn_22708743
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_227083002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������d
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������d
"
_user_specified_name
inputs/1
�
�
,__inference_Right_eye_layer_call_fn_22707839
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_227078242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������d
!
_user_specified_name	input_2
�
I
-__inference_activation_layer_call_fn_22709119

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_227080792
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_1_layer_call_and_return_conditional_losses_22709062

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_flatten_layer_call_and_return_conditional_losses_22707960

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������
 :S O
+
_output_shapes
:���������
 
 
_user_specified_nameinputs
�
�
+__inference_conv1d_3_layer_call_fn_22709255

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_227076632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������d	::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d	
 
_user_specified_nameinputs
�
�
+__inference_Left_eye_layer_call_fn_22707585
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_227075702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������d
!
_user_specified_name	input_1
�:
�
G__inference_Right_eye_layer_call_and_return_conditional_losses_22708973

inputs8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource8
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource
identity�X
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_1/Shape�
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack�
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1�
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/2�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape�
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:���������d	2
reshape_1/Reshape�
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_3/conv1d/ExpandDims/dim�
conv1d_3/conv1d/ExpandDims
ExpandDimsreshape_1/Reshape:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2
conv1d_3/conv1d/ExpandDims�
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim�
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d_3/conv1d/ExpandDims_1�
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
conv1d_3/conv1d�
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2
conv1d_3/conv1d/Squeeze�
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp�
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
conv1d_3/Relu�
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_4/conv1d/ExpandDims/dim�
conv1d_4/conv1d/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2
conv1d_4/conv1d/ExpandDims�
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dim�
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_4/conv1d/ExpandDims_1�
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d_4/conv1d�
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d_4/conv1d/Squeeze�
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_4/BiasAdd/ReadVariableOp�
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
conv1d_4/BiasAddw
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
conv1d_4/Relu�
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_5/conv1d/ExpandDims/dim�
conv1d_5/conv1d/ExpandDims
ExpandDimsconv1d_4/Relu:activations:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d_5/conv1d/ExpandDims�
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dim�
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_5/conv1d/ExpandDims_1�
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
conv1d_5/conv1d�
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2
conv1d_5/conv1d/Squeeze�
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_5/BiasAdd/ReadVariableOp�
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2
conv1d_5/BiasAddw
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:���������
2
conv1d_5/Relus
IdentityIdentityconv1d_5/Relu:activations:0*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d:::::::W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
a
E__inference_reshape_layer_call_and_return_conditional_losses_22707422

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape/shape/2�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������d	2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������d	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
F__inference_conv1d_5_layer_call_and_return_conditional_losses_22709296

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������
2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :::S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_conv1d_3_layer_call_and_return_conditional_losses_22707663

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������d	2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������/@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������/@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������/@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������/@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������d	:::S O
+
_output_shapes
:���������d	
 
_user_specified_nameinputs
�
�
,__inference_Right_eye_layer_call_fn_22707802
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_227077872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������d
!
_user_specified_name	input_2
�

*__inference_dense_1_layer_call_fn_22709071

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_227080062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_dense_layer_call_fn_22709051

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_227079792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�-
�
J__inference_functional_1_layer_call_and_return_conditional_losses_22708088
left	
right
left_eye_22707878
left_eye_22707880
left_eye_22707882
left_eye_22707884
left_eye_22707886
left_eye_22707888
right_eye_22707925
right_eye_22707927
right_eye_22707929
right_eye_22707931
right_eye_22707933
right_eye_22707935
dense_22707990
dense_22707992
dense_1_22708017
dense_1_22708019
dense_2_22708044
dense_2_22708046
dense_3_22708070
dense_3_22708072
identity�� Left_eye/StatefulPartitionedCall�!Right_eye/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
 Left_eye/StatefulPartitionedCallStatefulPartitionedCallleftleft_eye_22707878left_eye_22707880left_eye_22707882left_eye_22707884left_eye_22707886left_eye_22707888*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_227075702"
 Left_eye/StatefulPartitionedCall�
!Right_eye/StatefulPartitionedCallStatefulPartitionedCallrightright_eye_22707925right_eye_22707927right_eye_22707929right_eye_22707931right_eye_22707933right_eye_22707935*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_227077872#
!Right_eye/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall)Left_eye/StatefulPartitionedCall:output:0*Right_eye/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_227079452
concatenate/PartitionedCall�
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_227079602
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_22707990dense_22707992*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_227079792
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_22708017dense_1_22708019*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_227080062!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_22708044dense_2_22708046*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_227080332!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_22708070dense_3_22708072*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_227080592!
dense_3/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_227080792
activation/PartitionedCall�
IdentityIdentity#activation/PartitionedCall:output:0!^Left_eye/StatefulPartitionedCall"^Right_eye/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d::::::::::::::::::::2D
 Left_eye/StatefulPartitionedCall Left_eye/StatefulPartitionedCall2F
!Right_eye/StatefulPartitionedCall!Right_eye/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:U Q
/
_output_shapes
:���������d

_user_specified_nameLeft:VR
/
_output_shapes
:���������d

_user_specified_nameRight
�
�
E__inference_dense_2_layer_call_and_return_conditional_losses_22709082

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
G__inference_reshape_1_layer_call_and_return_conditional_losses_22709225

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape/shape/2�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������d	2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������d	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
F__inference_Left_eye_layer_call_and_return_conditional_losses_22707570

inputs
conv1d_22707554
conv1d_22707556
conv1d_1_22707559
conv1d_1_22707561
conv1d_2_22707564
conv1d_2_22707566
identity��conv1d/StatefulPartitionedCall� conv1d_1/StatefulPartitionedCall� conv1d_2/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_227074222
reshape/PartitionedCall�
conv1d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_22707554conv1d_22707556*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_227074462 
conv1d/StatefulPartitionedCall�
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_22707559conv1d_1_22707561*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_227074782"
 conv1d_1/StatefulPartitionedCall�
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_22707564conv1d_2_22707566*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_227075102"
 conv1d_2/StatefulPartitionedCall�
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
u
I__inference_concatenate_layer_call_and_return_conditional_losses_22709014
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:���������
 2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:���������
 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:���������
:���������
:U Q
+
_output_shapes
:���������

"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������

"
_user_specified_name
inputs/1
�
�
G__inference_Right_eye_layer_call_and_return_conditional_losses_22707787

inputs
conv1d_3_22707771
conv1d_3_22707773
conv1d_4_22707776
conv1d_4_22707778
conv1d_5_22707781
conv1d_5_22707783
identity�� conv1d_3/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_227076392
reshape_1/PartitionedCall�
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_3_22707771conv1d_3_22707773*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_227076632"
 conv1d_3/StatefulPartitionedCall�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_22707776conv1d_4_22707778*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_227076952"
 conv1d_4/StatefulPartitionedCall�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0conv1d_5_22707781conv1d_5_22707783*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_227077272"
 conv1d_5/StatefulPartitionedCall�
IdentityIdentity)conv1d_5/StatefulPartitionedCall:output:0!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
G__inference_Right_eye_layer_call_and_return_conditional_losses_22707824

inputs
conv1d_3_22707808
conv1d_3_22707810
conv1d_4_22707813
conv1d_4_22707815
conv1d_5_22707818
conv1d_5_22707820
identity�� conv1d_3/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_227076392
reshape_1/PartitionedCall�
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_3_22707808conv1d_3_22707810*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_227076632"
 conv1d_3/StatefulPartitionedCall�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_22707813conv1d_4_22707815*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_227076952"
 conv1d_4/StatefulPartitionedCall�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0conv1d_5_22707818conv1d_5_22707820*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_227077272"
 conv1d_5/StatefulPartitionedCall�
IdentityIdentity)conv1d_5/StatefulPartitionedCall:output:0!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
F__inference_Left_eye_layer_call_and_return_conditional_losses_22707527
input_1
conv1d_22707457
conv1d_22707459
conv1d_1_22707489
conv1d_1_22707491
conv1d_2_22707521
conv1d_2_22707523
identity��conv1d/StatefulPartitionedCall� conv1d_1/StatefulPartitionedCall� conv1d_2/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_227074222
reshape/PartitionedCall�
conv1d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_22707457conv1d_22707459*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_227074462 
conv1d/StatefulPartitionedCall�
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_22707489conv1d_1_22707491*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_227074782"
 conv1d_1/StatefulPartitionedCall�
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_22707521conv1d_2_22707523*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_227075102"
 conv1d_2/StatefulPartitionedCall�
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:X T
/
_output_shapes
:���������d
!
_user_specified_name	input_1
�-
�
J__inference_functional_1_layer_call_and_return_conditional_losses_22708142
left	
right
left_eye_22708092
left_eye_22708094
left_eye_22708096
left_eye_22708098
left_eye_22708100
left_eye_22708102
right_eye_22708105
right_eye_22708107
right_eye_22708109
right_eye_22708111
right_eye_22708113
right_eye_22708115
dense_22708120
dense_22708122
dense_1_22708125
dense_1_22708127
dense_2_22708130
dense_2_22708132
dense_3_22708135
dense_3_22708137
identity�� Left_eye/StatefulPartitionedCall�!Right_eye/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
 Left_eye/StatefulPartitionedCallStatefulPartitionedCallleftleft_eye_22708092left_eye_22708094left_eye_22708096left_eye_22708098left_eye_22708100left_eye_22708102*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_227076072"
 Left_eye/StatefulPartitionedCall�
!Right_eye/StatefulPartitionedCallStatefulPartitionedCallrightright_eye_22708105right_eye_22708107right_eye_22708109right_eye_22708111right_eye_22708113right_eye_22708115*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_227078242#
!Right_eye/StatefulPartitionedCall�
concatenate/PartitionedCallPartitionedCall)Left_eye/StatefulPartitionedCall:output:0*Right_eye/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_227079452
concatenate/PartitionedCall�
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_227079602
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_22708120dense_22708122*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_227079792
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_22708125dense_1_22708127*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_227080062!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_22708130dense_2_22708132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_227080332!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_22708135dense_3_22708137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_227080592!
dense_3/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_227080792
activation/PartitionedCall�
IdentityIdentity#activation/PartitionedCall:output:0!^Left_eye/StatefulPartitionedCall"^Right_eye/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d::::::::::::::::::::2D
 Left_eye/StatefulPartitionedCall Left_eye/StatefulPartitionedCall2F
!Right_eye/StatefulPartitionedCall!Right_eye/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:U Q
/
_output_shapes
:���������d

_user_specified_nameLeft:VR
/
_output_shapes
:���������d

_user_specified_nameRight
�
�
+__inference_Left_eye_layer_call_fn_22708875

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_227076072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
,__inference_Right_eye_layer_call_fn_22708990

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_227077872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
G__inference_Right_eye_layer_call_and_return_conditional_losses_22707764
input_2
conv1d_3_22707748
conv1d_3_22707750
conv1d_4_22707753
conv1d_4_22707755
conv1d_5_22707758
conv1d_5_22707760
identity�� conv1d_3/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_227076392
reshape_1/PartitionedCall�
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_3_22707748conv1d_3_22707750*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_227076632"
 conv1d_3/StatefulPartitionedCall�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_22707753conv1d_4_22707755*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_227076952"
 conv1d_4/StatefulPartitionedCall�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0conv1d_5_22707758conv1d_5_22707760*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_227077272"
 conv1d_5/StatefulPartitionedCall�
IdentityIdentity)conv1d_5/StatefulPartitionedCall:output:0!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������d::::::2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������d
!
_user_specified_name	input_2
�
�
&__inference_signature_wrapper_22708399
left	
right
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallleftrightunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *,
f'R%
#__inference__wrapped_model_227074052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
/
_output_shapes
:���������d

_user_specified_nameLeft:VR
/
_output_shapes
:���������d

_user_specified_nameRight
�
�
+__inference_conv1d_4_layer_call_fn_22709280

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_227076952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������/@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������/@
 
_user_specified_nameinputs
�
d
H__inference_activation_layer_call_and_return_conditional_losses_22709114

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
!__inference__traced_save_22709530
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop,
(savev2_conv1d_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_3_kernel_m_read_readvariableop3
/savev2_adam_conv1d_3_bias_m_read_readvariableop5
1savev2_adam_conv1d_4_kernel_m_read_readvariableop3
/savev2_adam_conv1d_4_bias_m_read_readvariableop5
1savev2_adam_conv1d_5_kernel_m_read_readvariableop3
/savev2_adam_conv1d_5_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop5
1savev2_adam_conv1d_3_kernel_v_read_readvariableop3
/savev2_adam_conv1d_3_bias_v_read_readvariableop5
1savev2_adam_conv1d_4_kernel_v_read_readvariableop3
/savev2_adam_conv1d_4_bias_v_read_readvariableop5
1savev2_adam_conv1d_5_kernel_v_read_readvariableop3
/savev2_adam_conv1d_5_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_6c7eba82d54e4954b9902e22a1db989f/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*�!
value�!B�!DB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*�
value�B�DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop(savev2_conv1d_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop1savev2_adam_conv1d_3_kernel_m_read_readvariableop/savev2_adam_conv1d_3_bias_m_read_readvariableop1savev2_adam_conv1d_4_kernel_m_read_readvariableop/savev2_adam_conv1d_4_bias_m_read_readvariableop1savev2_adam_conv1d_5_kernel_m_read_readvariableop/savev2_adam_conv1d_5_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop1savev2_adam_conv1d_3_kernel_v_read_readvariableop/savev2_adam_conv1d_3_bias_v_read_readvariableop1savev2_adam_conv1d_4_kernel_v_read_readvariableop/savev2_adam_conv1d_4_bias_v_read_readvariableop1savev2_adam_conv1d_5_kernel_v_read_readvariableop/savev2_adam_conv1d_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:
��:�:	�@:@:@:: : : : : :	@:@:@ : : ::	@:@:@ : : :: : :
��:�:
��:�:	�@:@:@::	@:@:@ : : ::	@:@:@ : : ::
��:�:
��:�:	�@:@:@::	@:@:@ : : ::	@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:	@: 

_output_shapes
:@:($
"
_output_shapes
:@ : 

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
::($
"
_output_shapes
:	@: 

_output_shapes
:@:($
"
_output_shapes
:@ : 

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:% !

_output_shapes
:	�@: !

_output_shapes
:@:$" 

_output_shapes

:@: #

_output_shapes
::($$
"
_output_shapes
:	@: %

_output_shapes
:@:(&$
"
_output_shapes
:@ : '

_output_shapes
: :(($
"
_output_shapes
: : )

_output_shapes
::(*$
"
_output_shapes
:	@: +

_output_shapes
:@:(,$
"
_output_shapes
:@ : -

_output_shapes
: :(.$
"
_output_shapes
: : /

_output_shapes
::&0"
 
_output_shapes
:
��:!1

_output_shapes	
:�:&2"
 
_output_shapes
:
��:!3

_output_shapes	
:�:%4!

_output_shapes
:	�@: 5

_output_shapes
:@:$6 

_output_shapes

:@: 7

_output_shapes
::(8$
"
_output_shapes
:	@: 9

_output_shapes
:@:(:$
"
_output_shapes
:@ : ;

_output_shapes
: :(<$
"
_output_shapes
: : =

_output_shapes
::(>$
"
_output_shapes
:	@: ?

_output_shapes
:@:(@$
"
_output_shapes
:@ : A

_output_shapes
: :(B$
"
_output_shapes
: : C

_output_shapes
::D

_output_shapes
: 
�
�
F__inference_conv1d_2_layer_call_and_return_conditional_losses_22709203

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:��������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������
*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������
2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :::S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_conv1d_1_layer_call_and_return_conditional_losses_22709178

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity�y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������/@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:��������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������/@:::S O
+
_output_shapes
:���������/@
 
_user_specified_nameinputs
�
�
/__inference_functional_1_layer_call_fn_22708697
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_227082002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������d:���������d::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������d
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������d
"
_user_specified_name
inputs/1
�
~
)__inference_conv1d_layer_call_fn_22709162

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_227074462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������d	::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d	
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
Left5
serving_default_Left:0���������d
?
Right6
serving_default_Right:0���������d>

activation0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
��
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
#_self_saveable_object_factories
	optimizer
loss

signatures
	variables
regularization_losses
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"А
_tf_keras_network��{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Left"}, "name": "Left", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Right"}, "name": "Right", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "Left_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv1d_2", 0, 0]]}, "name": "Left_eye", "inbound_nodes": [[["Left", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "Right_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["conv1d_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv1d_5", 0, 0]]}, "name": "Right_eye", "inbound_nodes": [[["Right", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["Left_eye", 1, 0, {}], ["Right_eye", 1, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "linear"}, "name": "activation", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}], "input_layers": [["Left", 0, 0], ["Right", 0, 0]], "output_layers": [["activation", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 100, 3, 3]}, {"class_name": "TensorShape", "items": [null, 100, 3, 3]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Left"}, "name": "Left", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Right"}, "name": "Right", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "Left_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv1d_2", 0, 0]]}, "name": "Left_eye", "inbound_nodes": [[["Left", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "Right_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["conv1d_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv1d_5", 0, 0]]}, "name": "Right_eye", "inbound_nodes": [[["Right", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["Left_eye", 1, 0, {}], ["Right_eye", 1, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "linear"}, "name": "activation", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}], "input_layers": [["Left", 0, 0], ["Right", 0, 0]], "output_layers": [["activation", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
#_self_saveable_object_factories"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "Left", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Left"}}
�
#_self_saveable_object_factories"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "Right", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Right"}}
�0
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�.
_tf_keras_network�.{"class_name": "Functional", "name": "Left_eye", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Left_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv1d_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 3, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Left_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv1d_2", 0, 0]]}}}
�1
 layer-0
!layer-1
"layer_with_weights-0
"layer-2
#layer_with_weights-1
#layer-3
$layer_with_weights-2
$layer-4
#%_self_saveable_object_factories
&	variables
'regularization_losses
(trainable_variables
)	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�.
_tf_keras_network�.{"class_name": "Functional", "name": "Right_eye", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Right_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["conv1d_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv1d_5", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 3, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Right_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["conv1d_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv1d_5", 0, 0]]}}}
�
#*_self_saveable_object_factories
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10, 16]}, {"class_name": "TensorShape", "items": [null, 10, 16]}]}
�
#/_self_saveable_object_factories
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

4kernel
5bias
#6_self_saveable_object_factories
7	variables
8regularization_losses
9trainable_variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 320}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 320]}}
�

;kernel
<bias
#=_self_saveable_object_factories
>	variables
?regularization_losses
@trainable_variables
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

Bkernel
Cbias
#D_self_saveable_object_factories
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�

Ikernel
Jbias
#K_self_saveable_object_factories
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
�
#P_self_saveable_object_factories
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "linear"}}
 "
trackable_dict_wrapper
�
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_rate4m�5m�;m�<m�Bm�Cm�Im�Jm�Zm�[m�\m�]m�^m�_m�`m�am�bm�cm�dm�em�4v�5v�;v�<v�Bv�Cv�Iv�Jv�Zv�[v�\v�]v�^v�_v�`v�av�bv�cv�dv�ev�"
	optimizer
 "
trackable_dict_wrapper
-
�serving_default"
signature_map
�
Z0
[1
\2
]3
^4
_5
`6
a7
b8
c9
d10
e11
412
513
;14
<15
B16
C17
I18
J19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Z0
[1
\2
]3
^4
_5
`6
a7
b8
c9
d10
e11
412
513
;14
<15
B16
C17
I18
J19"
trackable_list_wrapper
�
	variables
regularization_losses
fnon_trainable_variables
glayer_metrics
hlayer_regularization_losses

ilayers
trainable_variables
jmetrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
�
#k_self_saveable_object_factories"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�
#l_self_saveable_object_factories
m	variables
nregularization_losses
otrainable_variables
p	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}}
�


Zkernel
[bias
#q_self_saveable_object_factories
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 9]}}
�


\kernel
]bias
#v_self_saveable_object_factories
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 47, 64]}}
�


^kernel
_bias
#{_self_saveable_object_factories
|	variables
}regularization_losses
~trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 32]}}
 "
trackable_dict_wrapper
J
Z0
[1
\2
]3
^4
_5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
Z0
[1
\2
]3
^4
_5"
trackable_list_wrapper
�
	variables
regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
$�_self_saveable_object_factories"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}}
�


`kernel
abias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 9]}}
�


bkernel
cbias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 47, 64]}}
�


dkernel
ebias
$�_self_saveable_object_factories
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 32]}}
 "
trackable_dict_wrapper
J
`0
a1
b2
c3
d4
e5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
`0
a1
b2
c3
d4
e5"
trackable_list_wrapper
�
&	variables
'regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
(trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
+	variables
,regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
-trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0	variables
1regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
2trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :
��2dense/kernel
:�2
dense/bias
 "
trackable_dict_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
�
7	variables
8regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
9trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 
��2dense_1/kernel
:�2dense_1/bias
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
�
>	variables
?regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
@trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�@2dense_2/kernel
:@2dense_2/bias
 "
trackable_dict_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
�
E	variables
Fregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
Gtrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_3/kernel
:2dense_3/bias
 "
trackable_dict_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
�
L	variables
Mregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
Ntrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Q	variables
Rregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
Strainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
#:!	@2conv1d/kernel
:@2conv1d/bias
%:#@ 2conv1d_1/kernel
: 2conv1d_1/bias
%:# 2conv1d_2/kernel
:2conv1d_2/bias
%:#	@2conv1d_3/kernel
:@2conv1d_3/bias
%:#@ 2conv1d_4/kernel
: 2conv1d_4/bias
%:# 2conv1d_5/kernel
:2conv1d_5/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
m	variables
nregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
otrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
�
r	variables
sregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
ttrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
�
w	variables
xregularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
ytrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
�
|	variables
}regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
~trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
�trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
�
�	variables
�regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
�trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
�
�	variables
�regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
�trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
�
�	variables
�regularization_losses
�non_trainable_variables
�layer_metrics
 �layer_regularization_losses
�layers
�trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
 0
!1
"2
#3
$4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
%:#
��2Adam/dense/kernel/m
:�2Adam/dense/bias/m
':%
��2Adam/dense_1/kernel/m
 :�2Adam/dense_1/bias/m
&:$	�@2Adam/dense_2/kernel/m
:@2Adam/dense_2/bias/m
%:#@2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
(:&	@2Adam/conv1d/kernel/m
:@2Adam/conv1d/bias/m
*:(@ 2Adam/conv1d_1/kernel/m
 : 2Adam/conv1d_1/bias/m
*:( 2Adam/conv1d_2/kernel/m
 :2Adam/conv1d_2/bias/m
*:(	@2Adam/conv1d_3/kernel/m
 :@2Adam/conv1d_3/bias/m
*:(@ 2Adam/conv1d_4/kernel/m
 : 2Adam/conv1d_4/bias/m
*:( 2Adam/conv1d_5/kernel/m
 :2Adam/conv1d_5/bias/m
%:#
��2Adam/dense/kernel/v
:�2Adam/dense/bias/v
':%
��2Adam/dense_1/kernel/v
 :�2Adam/dense_1/bias/v
&:$	�@2Adam/dense_2/kernel/v
:@2Adam/dense_2/bias/v
%:#@2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
(:&	@2Adam/conv1d/kernel/v
:@2Adam/conv1d/bias/v
*:(@ 2Adam/conv1d_1/kernel/v
 : 2Adam/conv1d_1/bias/v
*:( 2Adam/conv1d_2/kernel/v
 :2Adam/conv1d_2/bias/v
*:(	@2Adam/conv1d_3/kernel/v
 :@2Adam/conv1d_3/bias/v
*:(@ 2Adam/conv1d_4/kernel/v
 : 2Adam/conv1d_4/bias/v
*:( 2Adam/conv1d_5/kernel/v
 :2Adam/conv1d_5/bias/v
�2�
J__inference_functional_1_layer_call_and_return_conditional_losses_22708525
J__inference_functional_1_layer_call_and_return_conditional_losses_22708651
J__inference_functional_1_layer_call_and_return_conditional_losses_22708088
J__inference_functional_1_layer_call_and_return_conditional_losses_22708142�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference__wrapped_model_22707405�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *Y�V
T�Q
&�#
Left���������d
'�$
Right���������d
�2�
/__inference_functional_1_layer_call_fn_22708743
/__inference_functional_1_layer_call_fn_22708243
/__inference_functional_1_layer_call_fn_22708343
/__inference_functional_1_layer_call_fn_22708697�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_Left_eye_layer_call_and_return_conditional_losses_22708792
F__inference_Left_eye_layer_call_and_return_conditional_losses_22708841
F__inference_Left_eye_layer_call_and_return_conditional_losses_22707527
F__inference_Left_eye_layer_call_and_return_conditional_losses_22707547�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_Left_eye_layer_call_fn_22707622
+__inference_Left_eye_layer_call_fn_22707585
+__inference_Left_eye_layer_call_fn_22708858
+__inference_Left_eye_layer_call_fn_22708875�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_Right_eye_layer_call_and_return_conditional_losses_22707764
G__inference_Right_eye_layer_call_and_return_conditional_losses_22708924
G__inference_Right_eye_layer_call_and_return_conditional_losses_22708973
G__inference_Right_eye_layer_call_and_return_conditional_losses_22707744�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_Right_eye_layer_call_fn_22707839
,__inference_Right_eye_layer_call_fn_22709007
,__inference_Right_eye_layer_call_fn_22708990
,__inference_Right_eye_layer_call_fn_22707802�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
I__inference_concatenate_layer_call_and_return_conditional_losses_22709014�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_concatenate_layer_call_fn_22709020�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_flatten_layer_call_and_return_conditional_losses_22709026�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_flatten_layer_call_fn_22709031�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_layer_call_and_return_conditional_losses_22709042�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_layer_call_fn_22709051�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_1_layer_call_and_return_conditional_losses_22709062�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_1_layer_call_fn_22709071�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_2_layer_call_and_return_conditional_losses_22709082�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_2_layer_call_fn_22709091�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_3_layer_call_and_return_conditional_losses_22709101�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_3_layer_call_fn_22709110�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_activation_layer_call_and_return_conditional_losses_22709114�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_activation_layer_call_fn_22709119�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
7B5
&__inference_signature_wrapper_22708399LeftRight
�2�
E__inference_reshape_layer_call_and_return_conditional_losses_22709132�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_reshape_layer_call_fn_22709137�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv1d_layer_call_and_return_conditional_losses_22709153�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv1d_layer_call_fn_22709162�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv1d_1_layer_call_and_return_conditional_losses_22709178�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv1d_1_layer_call_fn_22709187�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv1d_2_layer_call_and_return_conditional_losses_22709203�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv1d_2_layer_call_fn_22709212�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_reshape_1_layer_call_and_return_conditional_losses_22709225�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_reshape_1_layer_call_fn_22709230�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv1d_3_layer_call_and_return_conditional_losses_22709246�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv1d_3_layer_call_fn_22709255�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv1d_4_layer_call_and_return_conditional_losses_22709271�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv1d_4_layer_call_fn_22709280�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv1d_5_layer_call_and_return_conditional_losses_22709296�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv1d_5_layer_call_fn_22709305�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
F__inference_Left_eye_layer_call_and_return_conditional_losses_22707527uZ[\]^_@�=
6�3
)�&
input_1���������d
p

 
� ")�&
�
0���������

� �
F__inference_Left_eye_layer_call_and_return_conditional_losses_22707547uZ[\]^_@�=
6�3
)�&
input_1���������d
p 

 
� ")�&
�
0���������

� �
F__inference_Left_eye_layer_call_and_return_conditional_losses_22708792tZ[\]^_?�<
5�2
(�%
inputs���������d
p

 
� ")�&
�
0���������

� �
F__inference_Left_eye_layer_call_and_return_conditional_losses_22708841tZ[\]^_?�<
5�2
(�%
inputs���������d
p 

 
� ")�&
�
0���������

� �
+__inference_Left_eye_layer_call_fn_22707585hZ[\]^_@�=
6�3
)�&
input_1���������d
p

 
� "����������
�
+__inference_Left_eye_layer_call_fn_22707622hZ[\]^_@�=
6�3
)�&
input_1���������d
p 

 
� "����������
�
+__inference_Left_eye_layer_call_fn_22708858gZ[\]^_?�<
5�2
(�%
inputs���������d
p

 
� "����������
�
+__inference_Left_eye_layer_call_fn_22708875gZ[\]^_?�<
5�2
(�%
inputs���������d
p 

 
� "����������
�
G__inference_Right_eye_layer_call_and_return_conditional_losses_22707744u`abcde@�=
6�3
)�&
input_2���������d
p

 
� ")�&
�
0���������

� �
G__inference_Right_eye_layer_call_and_return_conditional_losses_22707764u`abcde@�=
6�3
)�&
input_2���������d
p 

 
� ")�&
�
0���������

� �
G__inference_Right_eye_layer_call_and_return_conditional_losses_22708924t`abcde?�<
5�2
(�%
inputs���������d
p

 
� ")�&
�
0���������

� �
G__inference_Right_eye_layer_call_and_return_conditional_losses_22708973t`abcde?�<
5�2
(�%
inputs���������d
p 

 
� ")�&
�
0���������

� �
,__inference_Right_eye_layer_call_fn_22707802h`abcde@�=
6�3
)�&
input_2���������d
p

 
� "����������
�
,__inference_Right_eye_layer_call_fn_22707839h`abcde@�=
6�3
)�&
input_2���������d
p 

 
� "����������
�
,__inference_Right_eye_layer_call_fn_22708990g`abcde?�<
5�2
(�%
inputs���������d
p

 
� "����������
�
,__inference_Right_eye_layer_call_fn_22709007g`abcde?�<
5�2
(�%
inputs���������d
p 

 
� "����������
�
#__inference__wrapped_model_22707405�Z[\]^_`abcde45;<BCIJc�`
Y�V
T�Q
&�#
Left���������d
'�$
Right���������d
� "7�4
2

activation$�!

activation����������
H__inference_activation_layer_call_and_return_conditional_losses_22709114X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� |
-__inference_activation_layer_call_fn_22709119K/�,
%�"
 �
inputs���������
� "�����������
I__inference_concatenate_layer_call_and_return_conditional_losses_22709014�b�_
X�U
S�P
&�#
inputs/0���������

&�#
inputs/1���������

� ")�&
�
0���������
 
� �
.__inference_concatenate_layer_call_fn_22709020�b�_
X�U
S�P
&�#
inputs/0���������

&�#
inputs/1���������

� "����������
 �
F__inference_conv1d_1_layer_call_and_return_conditional_losses_22709178d\]3�0
)�&
$�!
inputs���������/@
� ")�&
�
0��������� 
� �
+__inference_conv1d_1_layer_call_fn_22709187W\]3�0
)�&
$�!
inputs���������/@
� "���������� �
F__inference_conv1d_2_layer_call_and_return_conditional_losses_22709203d^_3�0
)�&
$�!
inputs��������� 
� ")�&
�
0���������

� �
+__inference_conv1d_2_layer_call_fn_22709212W^_3�0
)�&
$�!
inputs��������� 
� "����������
�
F__inference_conv1d_3_layer_call_and_return_conditional_losses_22709246d`a3�0
)�&
$�!
inputs���������d	
� ")�&
�
0���������/@
� �
+__inference_conv1d_3_layer_call_fn_22709255W`a3�0
)�&
$�!
inputs���������d	
� "����������/@�
F__inference_conv1d_4_layer_call_and_return_conditional_losses_22709271dbc3�0
)�&
$�!
inputs���������/@
� ")�&
�
0��������� 
� �
+__inference_conv1d_4_layer_call_fn_22709280Wbc3�0
)�&
$�!
inputs���������/@
� "���������� �
F__inference_conv1d_5_layer_call_and_return_conditional_losses_22709296dde3�0
)�&
$�!
inputs��������� 
� ")�&
�
0���������

� �
+__inference_conv1d_5_layer_call_fn_22709305Wde3�0
)�&
$�!
inputs��������� 
� "����������
�
D__inference_conv1d_layer_call_and_return_conditional_losses_22709153dZ[3�0
)�&
$�!
inputs���������d	
� ")�&
�
0���������/@
� �
)__inference_conv1d_layer_call_fn_22709162WZ[3�0
)�&
$�!
inputs���������d	
� "����������/@�
E__inference_dense_1_layer_call_and_return_conditional_losses_22709062^;<0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� 
*__inference_dense_1_layer_call_fn_22709071Q;<0�-
&�#
!�
inputs����������
� "������������
E__inference_dense_2_layer_call_and_return_conditional_losses_22709082]BC0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� ~
*__inference_dense_2_layer_call_fn_22709091PBC0�-
&�#
!�
inputs����������
� "����������@�
E__inference_dense_3_layer_call_and_return_conditional_losses_22709101\IJ/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� }
*__inference_dense_3_layer_call_fn_22709110OIJ/�,
%�"
 �
inputs���������@
� "�����������
C__inference_dense_layer_call_and_return_conditional_losses_22709042^450�-
&�#
!�
inputs����������
� "&�#
�
0����������
� }
(__inference_dense_layer_call_fn_22709051Q450�-
&�#
!�
inputs����������
� "������������
E__inference_flatten_layer_call_and_return_conditional_losses_22709026]3�0
)�&
$�!
inputs���������
 
� "&�#
�
0����������
� ~
*__inference_flatten_layer_call_fn_22709031P3�0
)�&
$�!
inputs���������
 
� "������������
J__inference_functional_1_layer_call_and_return_conditional_losses_22708088�Z[\]^_`abcde45;<BCIJk�h
a�^
T�Q
&�#
Left���������d
'�$
Right���������d
p

 
� "%�"
�
0���������
� �
J__inference_functional_1_layer_call_and_return_conditional_losses_22708142�Z[\]^_`abcde45;<BCIJk�h
a�^
T�Q
&�#
Left���������d
'�$
Right���������d
p 

 
� "%�"
�
0���������
� �
J__inference_functional_1_layer_call_and_return_conditional_losses_22708525�Z[\]^_`abcde45;<BCIJr�o
h�e
[�X
*�'
inputs/0���������d
*�'
inputs/1���������d
p

 
� "%�"
�
0���������
� �
J__inference_functional_1_layer_call_and_return_conditional_losses_22708651�Z[\]^_`abcde45;<BCIJr�o
h�e
[�X
*�'
inputs/0���������d
*�'
inputs/1���������d
p 

 
� "%�"
�
0���������
� �
/__inference_functional_1_layer_call_fn_22708243�Z[\]^_`abcde45;<BCIJk�h
a�^
T�Q
&�#
Left���������d
'�$
Right���������d
p

 
� "�����������
/__inference_functional_1_layer_call_fn_22708343�Z[\]^_`abcde45;<BCIJk�h
a�^
T�Q
&�#
Left���������d
'�$
Right���������d
p 

 
� "�����������
/__inference_functional_1_layer_call_fn_22708697�Z[\]^_`abcde45;<BCIJr�o
h�e
[�X
*�'
inputs/0���������d
*�'
inputs/1���������d
p

 
� "�����������
/__inference_functional_1_layer_call_fn_22708743�Z[\]^_`abcde45;<BCIJr�o
h�e
[�X
*�'
inputs/0���������d
*�'
inputs/1���������d
p 

 
� "�����������
G__inference_reshape_1_layer_call_and_return_conditional_losses_22709225d7�4
-�*
(�%
inputs���������d
� ")�&
�
0���������d	
� �
,__inference_reshape_1_layer_call_fn_22709230W7�4
-�*
(�%
inputs���������d
� "����������d	�
E__inference_reshape_layer_call_and_return_conditional_losses_22709132d7�4
-�*
(�%
inputs���������d
� ")�&
�
0���������d	
� �
*__inference_reshape_layer_call_fn_22709137W7�4
-�*
(�%
inputs���������d
� "����������d	�
&__inference_signature_wrapper_22708399�Z[\]^_`abcde45;<BCIJo�l
� 
e�b
.
Left&�#
Left���������d
0
Right'�$
Right���������d"7�4
2

activation$�!

activation���������