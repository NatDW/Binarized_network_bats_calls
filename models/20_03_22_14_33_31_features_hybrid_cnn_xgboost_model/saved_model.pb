Ã&
«ý
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
dtypetype
¾
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02v1.12.1-25073-g2c5e22190c8äÈ 

quant_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*$
shared_namequant_conv2d/kernel

'quant_conv2d/kernel/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel*&
_output_shapes
:8*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:8*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:8*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:8*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:8*
dtype0

quant_conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*&
shared_namequant_conv2d_1/kernel

)quant_conv2d_1/kernel/Read/ReadVariableOpReadVariableOpquant_conv2d_1/kernel*&
_output_shapes
:88*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:8*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:8*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:8*
dtype0
¢
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:8*
dtype0

quant_conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*&
shared_namequant_conv2d_2/kernel

)quant_conv2d_2/kernel/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel*&
_output_shapes
:88*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:8*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:8*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:8*
dtype0
¢
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:8*
dtype0

quant_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_namequant_dense/kernel
{
&quant_dense/kernel/Read/ReadVariableOpReadVariableOpquant_dense/kernel* 
_output_shapes
:
*
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:*
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:*
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:*
dtype0

quant_dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_namequant_dense_1/kernel

(quant_dense_1/kernel/Read/ReadVariableOpReadVariableOpquant_dense_1/kernel* 
_output_shapes
:
*
dtype0

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:*
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:*
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:*
dtype0

quant_dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_namequant_dense_2/kernel
~
(quant_dense_2/kernel/Read/ReadVariableOpReadVariableOpquant_dense_2/kernel*
_output_shapes
:	*
dtype0

batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma

/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:*
dtype0

batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta

.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0

!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean

5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
¢
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance

9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/quant_conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*+
shared_nameAdam/quant_conv2d/kernel/m

.Adam/quant_conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d/kernel/m*&
_output_shapes
:8*
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:8*
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:8*
dtype0

Adam/quant_conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*-
shared_nameAdam/quant_conv2d_1/kernel/m

0Adam/quant_conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_1/kernel/m*&
_output_shapes
:88*
dtype0

"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"Adam/batch_normalization_1/gamma/m

6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:8*
dtype0

!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!Adam/batch_normalization_1/beta/m

5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:8*
dtype0

Adam/quant_conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*-
shared_nameAdam/quant_conv2d_2/kernel/m

0Adam/quant_conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_2/kernel/m*&
_output_shapes
:88*
dtype0

"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"Adam/batch_normalization_2/gamma/m

6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:8*
dtype0

!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!Adam/batch_normalization_2/beta/m

5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:8*
dtype0

Adam/quant_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameAdam/quant_dense/kernel/m

-Adam/quant_dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_dense/kernel/m* 
_output_shapes
:
*
dtype0

"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/m

6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/m

5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes	
:*
dtype0

Adam/quant_dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameAdam/quant_dense_1/kernel/m

/Adam/quant_dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_dense_1/kernel/m* 
_output_shapes
:
*
dtype0

"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/m

6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/m

5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes	
:*
dtype0

Adam/quant_dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameAdam/quant_dense_2/kernel/m

/Adam/quant_dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_dense_2/kernel/m*
_output_shapes
:	*
dtype0

"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/m

6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
:*
dtype0

!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/m

5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
:*
dtype0

Adam/quant_conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*+
shared_nameAdam/quant_conv2d/kernel/v

.Adam/quant_conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d/kernel/v*&
_output_shapes
:8*
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:8*
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:8*
dtype0

Adam/quant_conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*-
shared_nameAdam/quant_conv2d_1/kernel/v

0Adam/quant_conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_1/kernel/v*&
_output_shapes
:88*
dtype0

"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"Adam/batch_normalization_1/gamma/v

6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:8*
dtype0

!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!Adam/batch_normalization_1/beta/v

5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:8*
dtype0

Adam/quant_conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*-
shared_nameAdam/quant_conv2d_2/kernel/v

0Adam/quant_conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_2/kernel/v*&
_output_shapes
:88*
dtype0

"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"Adam/batch_normalization_2/gamma/v

6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:8*
dtype0

!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!Adam/batch_normalization_2/beta/v

5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:8*
dtype0

Adam/quant_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameAdam/quant_dense/kernel/v

-Adam/quant_dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_dense/kernel/v* 
_output_shapes
:
*
dtype0

"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/v

6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/v

5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes	
:*
dtype0

Adam/quant_dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameAdam/quant_dense_1/kernel/v

/Adam/quant_dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_dense_1/kernel/v* 
_output_shapes
:
*
dtype0

"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/v

6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/v

5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes	
:*
dtype0

Adam/quant_dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameAdam/quant_dense_2/kernel/v

/Adam/quant_dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_dense_2/kernel/v*
_output_shapes
:	*
dtype0

"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/v

6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
:*
dtype0

!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/v

5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
÷
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*±
value¦B¢ B

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
t
kernel_quantizer

kernel
trainable_variables
	variables
regularization_losses
	keras_api

axis
	gamma
 beta
!moving_mean
"moving_variance
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api

+kernel_quantizer
,input_quantizer

-kernel
.trainable_variables
/	variables
0regularization_losses
1	keras_api

2axis
	3gamma
4beta
5moving_mean
6moving_variance
7trainable_variables
8	variables
9regularization_losses
:	keras_api
R
;trainable_variables
<	variables
=regularization_losses
>	keras_api

?kernel_quantizer
@input_quantizer

Akernel
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api

Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
R
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
R
Strainable_variables
T	variables
Uregularization_losses
V	keras_api

Wkernel_quantizer
Xinput_quantizer

Ykernel
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api

^axis
	_gamma
`beta
amoving_mean
bmoving_variance
ctrainable_variables
d	variables
eregularization_losses
f	keras_api

gkernel_quantizer
hinput_quantizer

ikernel
jtrainable_variables
k	variables
lregularization_losses
m	keras_api

naxis
	ogamma
pbeta
qmoving_mean
rmoving_variance
strainable_variables
t	variables
uregularization_losses
v	keras_api

wkernel_quantizer
xinput_quantizer

ykernel
ztrainable_variables
{	variables
|regularization_losses
}	keras_api

~axis
	gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
¯
	iter
beta_1
beta_2

decay
learning_ratemÎmÏ mÐ-mÑ3mÒ4mÓAmÔGmÕHmÖYm×_mØ`mÙimÚomÛpmÜymÝmÞ	mßvàvá vâ-vã3vä4våAvæGvçHvèYvé_vê`vëivìovípvîyvïvð	vñ

0
1
 2
-3
34
45
A6
G7
H8
Y9
_10
`11
i12
o13
p14
y15
16
17
é
0
1
 2
!3
"4
-5
36
47
58
69
A10
G11
H12
I13
J14
Y15
_16
`17
a18
b19
i20
o21
p22
q23
r24
y25
26
27
28
29
 

 layer_regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
	variables
regularization_losses
 
l
_custom_metrics
trainable_variables
	variables
regularization_losses
	keras_api
_]
VARIABLE_VALUEquant_conv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 

 layer_regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
	variables
regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
!2
"3
 

 layer_regularization_losses
#trainable_variables
metrics
non_trainable_variables
 layers
$	variables
%regularization_losses
 
 
 

 ¡layer_regularization_losses
'trainable_variables
¢metrics
£non_trainable_variables
¤layers
(	variables
)regularization_losses
l
¥_custom_metrics
¦trainable_variables
§	variables
¨regularization_losses
©	keras_api
V
ªtrainable_variables
«	variables
¬regularization_losses
­	keras_api
a_
VARIABLE_VALUEquant_conv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

-0

-0
 

 ®layer_regularization_losses
.trainable_variables
¯metrics
°non_trainable_variables
±layers
/	variables
0regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
52
63
 

 ²layer_regularization_losses
7trainable_variables
³metrics
´non_trainable_variables
µlayers
8	variables
9regularization_losses
 
 
 

 ¶layer_regularization_losses
;trainable_variables
·metrics
¸non_trainable_variables
¹layers
<	variables
=regularization_losses
l
º_custom_metrics
»trainable_variables
¼	variables
½regularization_losses
¾	keras_api
V
¿trainable_variables
À	variables
Áregularization_losses
Â	keras_api
a_
VARIABLE_VALUEquant_conv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

A0

A0
 

 Ãlayer_regularization_losses
Btrainable_variables
Ämetrics
Ånon_trainable_variables
Ælayers
C	variables
Dregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

G0
H1

G0
H1
I2
J3
 

 Çlayer_regularization_losses
Ktrainable_variables
Èmetrics
Énon_trainable_variables
Êlayers
L	variables
Mregularization_losses
 
 
 

 Ëlayer_regularization_losses
Otrainable_variables
Ìmetrics
Ínon_trainable_variables
Îlayers
P	variables
Qregularization_losses
 
 
 

 Ïlayer_regularization_losses
Strainable_variables
Ðmetrics
Ñnon_trainable_variables
Òlayers
T	variables
Uregularization_losses
l
Ó_custom_metrics
Ôtrainable_variables
Õ	variables
Öregularization_losses
×	keras_api
V
Øtrainable_variables
Ù	variables
Úregularization_losses
Û	keras_api
^\
VARIABLE_VALUEquant_dense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE

Y0

Y0
 

 Ülayer_regularization_losses
Ztrainable_variables
Ýmetrics
Þnon_trainable_variables
ßlayers
[	variables
\regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

_0
`1
a2
b3
 

 àlayer_regularization_losses
ctrainable_variables
ámetrics
ânon_trainable_variables
ãlayers
d	variables
eregularization_losses
l
ä_custom_metrics
åtrainable_variables
æ	variables
çregularization_losses
è	keras_api
V
étrainable_variables
ê	variables
ëregularization_losses
ì	keras_api
`^
VARIABLE_VALUEquant_dense_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE

i0

i0
 

 ílayer_regularization_losses
jtrainable_variables
îmetrics
ïnon_trainable_variables
ðlayers
k	variables
lregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

o0
p1

o0
p1
q2
r3
 

 ñlayer_regularization_losses
strainable_variables
òmetrics
ónon_trainable_variables
ôlayers
t	variables
uregularization_losses
l
õ_custom_metrics
ötrainable_variables
÷	variables
øregularization_losses
ù	keras_api
V
útrainable_variables
û	variables
üregularization_losses
ý	keras_api
a_
VARIABLE_VALUEquant_dense_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE

y0

y0
 

 þlayer_regularization_losses
ztrainable_variables
ÿmetrics
non_trainable_variables
layers
{	variables
|regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
2
3
 
¡
 layer_regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
	variables
regularization_losses
 
 
 
¡
 layer_regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
	variables
regularization_losses
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
 

0
1
X
!0
"1
52
63
I4
J5
a6
b7
q8
r9
10
11
~
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
11
12
13
14
15
16
 
 
 
 
¡
 layer_regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
	variables
regularization_losses
 
 
 

0
 
 

!0
"1
 
 
 
 
 
 
 
 
 
¡
 layer_regularization_losses
¦trainable_variables
metrics
non_trainable_variables
layers
§	variables
¨regularization_losses
 
 
 
¡
 layer_regularization_losses
ªtrainable_variables
metrics
non_trainable_variables
layers
«	variables
¬regularization_losses
 
 
 

+0
,1
 
 

50
61
 
 
 
 
 
 
 
 
 
¡
 layer_regularization_losses
»trainable_variables
metrics
non_trainable_variables
layers
¼	variables
½regularization_losses
 
 
 
¡
 layer_regularization_losses
¿trainable_variables
metrics
non_trainable_variables
layers
À	variables
Áregularization_losses
 
 
 

?0
@1
 
 

I0
J1
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
¡
  layer_regularization_losses
Ôtrainable_variables
¡metrics
¢non_trainable_variables
£layers
Õ	variables
Öregularization_losses
 
 
 
¡
 ¤layer_regularization_losses
Øtrainable_variables
¥metrics
¦non_trainable_variables
§layers
Ù	variables
Úregularization_losses
 
 
 

W0
X1
 
 

a0
b1
 
 
 
 
 
¡
 ¨layer_regularization_losses
åtrainable_variables
©metrics
ªnon_trainable_variables
«layers
æ	variables
çregularization_losses
 
 
 
¡
 ¬layer_regularization_losses
étrainable_variables
­metrics
®non_trainable_variables
¯layers
ê	variables
ëregularization_losses
 
 
 

g0
h1
 
 

q0
r1
 
 
 
 
 
¡
 °layer_regularization_losses
ötrainable_variables
±metrics
²non_trainable_variables
³layers
÷	variables
øregularization_losses
 
 
 
¡
 ´layer_regularization_losses
útrainable_variables
µmetrics
¶non_trainable_variables
·layers
û	variables
üregularization_losses
 
 
 

w0
x1
 
 

0
1
 
 
 
 
 


¸total

¹count
º
_fn_kwargs
»trainable_variables
¼	variables
½regularization_losses
¾	keras_api


¿total

Àcount
Á
_fn_kwargs
Âtrainable_variables
Ã	variables
Äregularization_losses
Å	keras_api
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
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

¸0
¹1
 
¡
 Ælayer_regularization_losses
»trainable_variables
Çmetrics
Ènon_trainable_variables
Élayers
¼	variables
½regularization_losses
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

¿0
À1
 
¡
 Êlayer_regularization_losses
Âtrainable_variables
Ëmetrics
Ìnon_trainable_variables
Ílayers
Ã	variables
Äregularization_losses
 
 

¸0
¹1
 
 
 

¿0
À1
 

VARIABLE_VALUEAdam/quant_conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_conv2d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_conv2d_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_dense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_dense_1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_dense_2/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_5/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_conv2d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_conv2d_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_dense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_dense_1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/quant_dense_2/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_5/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

"serving_default_quant_conv2d_inputPlaceholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿ
¥	
StatefulPartitionedCallStatefulPartitionedCall"serving_default_quant_conv2d_inputquant_conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancequant_conv2d_1/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancequant_conv2d_2/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancequant_dense/kernel%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betaquant_dense_1/kernel%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betaquant_dense_2/kernel%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/beta**
Tin#
!2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_183502
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'quant_conv2d/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp)quant_conv2d_1/kernel/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp)quant_conv2d_2/kernel/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp&quant_dense/kernel/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp(quant_dense_1/kernel/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp(quant_dense_2/kernel/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/quant_conv2d/kernel/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp0Adam/quant_conv2d_1/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp0Adam/quant_conv2d_2/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp-Adam/quant_dense/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp/Adam/quant_dense_1/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp/Adam/quant_dense_2/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp.Adam/quant_conv2d/kernel/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp0Adam/quant_conv2d_1/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp0Adam/quant_conv2d_2/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp-Adam/quant_dense/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp/Adam/quant_dense_1/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp/Adam/quant_dense_2/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_185515
¸
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamequant_conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancequant_conv2d_1/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancequant_conv2d_2/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancequant_dense/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancequant_dense_1/kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancequant_dense_2/kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/quant_conv2d/kernel/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/quant_conv2d_1/kernel/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/quant_conv2d_2/kernel/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/quant_dense/kernel/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/quant_dense_1/kernel/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/quant_dense_2/kernel/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/quant_conv2d/kernel/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/quant_conv2d_1/kernel/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/quant_conv2d_2/kernel/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/quant_dense/kernel/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/quant_dense_1/kernel/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/quant_dense_2/kernel/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/v*W
TinP
N2L*
Tout
2*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_185752Ã

¢
G__inference_quant_dense_layer_call_and_return_conditional_losses_184732

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOpe
ste_sign_6/SignSigninputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_6/Signi
ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
ste_sign_6/add/y
ste_sign_6/addAddV2ste_sign_6/Sign:y:0ste_sign_6/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_6/addu
ste_sign_6/Sign_1Signste_sign_6/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_6/Sign_1
ste_sign_6/IdentityIdentityste_sign_6/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_6/IdentityÑ
ste_sign_6/IdentityN	IdentityNste_sign_6/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-184712*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
ste_sign_6/IdentityN
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOp
MatMul/ste_sign_5/SignSignMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_5/Signw
MatMul/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
MatMul/ste_sign_5/add/y 
MatMul/ste_sign_5/addAddV2MatMul/ste_sign_5/Sign:y:0 MatMul/ste_sign_5/add/y:output:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_5/add
MatMul/ste_sign_5/Sign_1SignMatMul/ste_sign_5/add:z:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_5/Sign_1
MatMul/ste_sign_5/IdentityIdentityMatMul/ste_sign_5/Sign_1:y:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_5/Identityí
MatMul/ste_sign_5/IdentityN	IdentityNMatMul/ste_sign_5/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-184722*,
_output_shapes
:
:
2
MatMul/ste_sign_5/IdentityN
MatMulMatMulste_sign_6/IdentityN:output:0$MatMul/ste_sign_5/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_181920

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ0
È
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_182183

inputs
assignmovingavg_182158
assignmovingavg_1_182164)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/182158*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_182158*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpÄ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/182158*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/182158*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_182158AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/182158*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/182164*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_182164*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÎ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182164*
_output_shapes	
:2
AssignMovingAvg_1/subÅ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182164*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_182164AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/182164*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1´
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Å	
d
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_185227

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
add/yy
addAddV2Sign:y:0add/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identityâ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-185218*n
_output_shapes\
Z:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
	IdentityN

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
Ñ
G
+__inference_ste_sign_3_layer_call_fn_185249

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*&
_output_shapes
:88*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_1817552
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:882

Identity"
identityIdentity:output:0*%
_input_shapes
:88:N J
&
_output_shapes
:88
 
_user_specified_nameinputs


Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_182067

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¾ý

F__inference_sequential_layer_call_and_return_conditional_losses_183813

inputs/
+quant_conv2d_conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource.
*batch_normalization_assignmovingavg_1835320
,batch_normalization_assignmovingavg_1_1835391
-quant_conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resource0
,batch_normalization_1_assignmovingavg_1835822
.batch_normalization_1_assignmovingavg_1_1835891
-quant_conv2d_2_conv2d_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resource0
,batch_normalization_2_assignmovingavg_1836322
.batch_normalization_2_assignmovingavg_1_183639.
*quant_dense_matmul_readvariableop_resource0
,batch_normalization_3_assignmovingavg_1836792
.batch_normalization_3_assignmovingavg_1_183685?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource0
,quant_dense_1_matmul_readvariableop_resource0
,batch_normalization_4_assignmovingavg_1837332
.batch_normalization_4_assignmovingavg_1_183739?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource0
,quant_dense_2_matmul_readvariableop_resource0
,batch_normalization_5_assignmovingavg_1837872
.batch_normalization_5_assignmovingavg_1_183793?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource
identity¢7batch_normalization/AssignMovingAvg/AssignSubVariableOp¢2batch_normalization/AssignMovingAvg/ReadVariableOp¢9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp¢4batch_normalization/AssignMovingAvg_1/ReadVariableOp¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_1/AssignMovingAvg/ReadVariableOp¢;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_2/AssignMovingAvg/ReadVariableOp¢;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_3/AssignMovingAvg/ReadVariableOp¢;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_3/batchnorm/ReadVariableOp¢2batch_normalization_3/batchnorm/mul/ReadVariableOp¢9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_4/AssignMovingAvg/ReadVariableOp¢;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_5/AssignMovingAvg/ReadVariableOp¢;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢"quant_conv2d/Conv2D/ReadVariableOp¢$quant_conv2d_1/Conv2D/ReadVariableOp¢$quant_conv2d_2/Conv2D/ReadVariableOp¢!quant_dense/MatMul/ReadVariableOp¢#quant_dense_1/MatMul/ReadVariableOp¢#quant_dense_2/MatMul/ReadVariableOp¼
"quant_conv2d/Conv2D/ReadVariableOpReadVariableOp+quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02$
"quant_conv2d/Conv2D/ReadVariableOp«
!quant_conv2d/Conv2D/ste_sign/SignSign*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:82#
!quant_conv2d/Conv2D/ste_sign/Sign
"quant_conv2d/Conv2D/ste_sign/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2$
"quant_conv2d/Conv2D/ste_sign/add/yÒ
 quant_conv2d/Conv2D/ste_sign/addAddV2%quant_conv2d/Conv2D/ste_sign/Sign:y:0+quant_conv2d/Conv2D/ste_sign/add/y:output:0*
T0*&
_output_shapes
:82"
 quant_conv2d/Conv2D/ste_sign/add©
#quant_conv2d/Conv2D/ste_sign/Sign_1Sign$quant_conv2d/Conv2D/ste_sign/add:z:0*
T0*&
_output_shapes
:82%
#quant_conv2d/Conv2D/ste_sign/Sign_1´
%quant_conv2d/Conv2D/ste_sign/IdentityIdentity'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*
T0*&
_output_shapes
:82'
%quant_conv2d/Conv2D/ste_sign/Identity§
&quant_conv2d/Conv2D/ste_sign/IdentityN	IdentityN'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183507*8
_output_shapes&
$:8:82(
&quant_conv2d/Conv2D/ste_sign/IdentityNÐ
quant_conv2d/Conv2DConv2Dinputs/quant_conv2d/Conv2D/ste_sign/IdentityN:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
paddingSAME*
strides
2
quant_conv2d/Conv2D
 batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 batch_normalization/LogicalAnd/x
 batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 batch_normalization/LogicalAnd/y¼
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2 
batch_normalization/LogicalAnd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype02&
$batch_normalization/ReadVariableOp_1y
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization/Const}
batch_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization/Const_1
$batch_normalization/FusedBatchNormV3FusedBatchNormV3quant_conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0"batch_normalization/Const:output:0$batch_normalization/Const_1:output:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:2&
$batch_normalization/FusedBatchNormV3
batch_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization/Const_2Ú
)batch_normalization/AssignMovingAvg/sub/xConst*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/183532*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)batch_normalization/AssignMovingAvg/sub/x
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/183532*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/subÏ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_assignmovingavg_183532*
_output_shapes
:8*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp°
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/183532*
_output_shapes
:82+
)batch_normalization/AssignMovingAvg/sub_1
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/183532*
_output_shapes
:82)
'batch_normalization/AssignMovingAvg/mulù
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_assignmovingavg_183532+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/183532*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpà
+batch_normalization/AssignMovingAvg_1/sub/xConst*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/183539*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization/AssignMovingAvg_1/sub/x
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/183539*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/subÕ
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_assignmovingavg_1_183539*
_output_shapes
:8*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp¼
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/183539*
_output_shapes
:82-
+batch_normalization/AssignMovingAvg_1/sub_1£
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/183539*
_output_shapes
:82+
)batch_normalization/AssignMovingAvg_1/mul
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_assignmovingavg_1_183539-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/183539*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpÐ
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool¢
quant_conv2d_1/ste_sign_2/SignSignmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82 
quant_conv2d_1/ste_sign_2/Sign
quant_conv2d_1/ste_sign_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2!
quant_conv2d_1/ste_sign_2/add/yÏ
quant_conv2d_1/ste_sign_2/addAddV2"quant_conv2d_1/ste_sign_2/Sign:y:0(quant_conv2d_1/ste_sign_2/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82
quant_conv2d_1/ste_sign_2/add©
 quant_conv2d_1/ste_sign_2/Sign_1Sign!quant_conv2d_1/ste_sign_2/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82"
 quant_conv2d_1/ste_sign_2/Sign_1´
"quant_conv2d_1/ste_sign_2/IdentityIdentity$quant_conv2d_1/ste_sign_2/Sign_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82$
"quant_conv2d_1/ste_sign_2/Identity¤
#quant_conv2d_1/ste_sign_2/IdentityN	IdentityN$quant_conv2d_1/ste_sign_2/Sign_1:y:0max_pooling2d/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-183547*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿA
8:ÿÿÿÿÿÿÿÿÿA
82%
#quant_conv2d_1/ste_sign_2/IdentityNÂ
$quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_1/Conv2D/ReadVariableOpµ
%quant_conv2d_1/Conv2D/ste_sign_1/SignSign,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:882'
%quant_conv2d_1/Conv2D/ste_sign_1/Sign
&quant_conv2d_1/Conv2D/ste_sign_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2(
&quant_conv2d_1/Conv2D/ste_sign_1/add/yâ
$quant_conv2d_1/Conv2D/ste_sign_1/addAddV2)quant_conv2d_1/Conv2D/ste_sign_1/Sign:y:0/quant_conv2d_1/Conv2D/ste_sign_1/add/y:output:0*
T0*&
_output_shapes
:882&
$quant_conv2d_1/Conv2D/ste_sign_1/addµ
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1Sign(quant_conv2d_1/Conv2D/ste_sign_1/add:z:0*
T0*&
_output_shapes
:882)
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1À
)quant_conv2d_1/Conv2D/ste_sign_1/IdentityIdentity+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0*
T0*&
_output_shapes
:882+
)quant_conv2d_1/Conv2D/ste_sign_1/Identityµ
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityN	IdentityN+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183557*8
_output_shapes&
$:88:882,
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityNý
quant_conv2d_1/Conv2DConv2D,quant_conv2d_1/ste_sign_2/IdentityN:output:03quant_conv2d_1/Conv2D/ste_sign_1/IdentityN:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*
paddingSAME*
strides
2
quant_conv2d_1/Conv2D
"batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_1/LogicalAnd/x
"batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_1/LogicalAnd/yÄ
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_1/LogicalAnd¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:8*
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:8*
dtype02(
&batch_normalization_1/ReadVariableOp_1}
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_1/Const
batch_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_1/Const_1¡
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3quant_conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0$batch_normalization_1/Const:output:0&batch_normalization_1/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿA
8:8:8:8:8:*
epsilon%o:2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_1/Const_2à
+batch_normalization_1/AssignMovingAvg/sub/xConst*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/183582*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_1/AssignMovingAvg/sub/x
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/183582*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/subÕ
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_1_assignmovingavg_183582*
_output_shapes
:8*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpº
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/183582*
_output_shapes
:82-
+batch_normalization_1/AssignMovingAvg/sub_1£
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/183582*
_output_shapes
:82+
)batch_normalization_1/AssignMovingAvg/mul
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_1_assignmovingavg_183582-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/183582*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpæ
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/183589*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_1/AssignMovingAvg_1/sub/x¥
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/183589*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/subÛ
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_1_assignmovingavg_1_183589*
_output_shapes
:8*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpÆ
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/183589*
_output_shapes
:82/
-batch_normalization_1/AssignMovingAvg_1/sub_1­
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/183589*
_output_shapes
:82-
+batch_normalization_1/AssignMovingAvg_1/mul
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_1_assignmovingavg_1_183589/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/183589*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpÖ
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool¤
quant_conv2d_2/ste_sign_4/SignSign max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82 
quant_conv2d_2/ste_sign_4/Sign
quant_conv2d_2/ste_sign_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2!
quant_conv2d_2/ste_sign_4/add/yÏ
quant_conv2d_2/ste_sign_4/addAddV2"quant_conv2d_2/ste_sign_4/Sign:y:0(quant_conv2d_2/ste_sign_4/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82
quant_conv2d_2/ste_sign_4/add©
 quant_conv2d_2/ste_sign_4/Sign_1Sign!quant_conv2d_2/ste_sign_4/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82"
 quant_conv2d_2/ste_sign_4/Sign_1´
"quant_conv2d_2/ste_sign_4/IdentityIdentity$quant_conv2d_2/ste_sign_4/Sign_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82$
"quant_conv2d_2/ste_sign_4/Identity¦
#quant_conv2d_2/ste_sign_4/IdentityN	IdentityN$quant_conv2d_2/ste_sign_4/Sign_1:y:0 max_pooling2d_1/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-183597*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ 8:ÿÿÿÿÿÿÿÿÿ 82%
#quant_conv2d_2/ste_sign_4/IdentityNÂ
$quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_2/Conv2D/ReadVariableOpµ
%quant_conv2d_2/Conv2D/ste_sign_3/SignSign,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:882'
%quant_conv2d_2/Conv2D/ste_sign_3/Sign
&quant_conv2d_2/Conv2D/ste_sign_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2(
&quant_conv2d_2/Conv2D/ste_sign_3/add/yâ
$quant_conv2d_2/Conv2D/ste_sign_3/addAddV2)quant_conv2d_2/Conv2D/ste_sign_3/Sign:y:0/quant_conv2d_2/Conv2D/ste_sign_3/add/y:output:0*
T0*&
_output_shapes
:882&
$quant_conv2d_2/Conv2D/ste_sign_3/addµ
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1Sign(quant_conv2d_2/Conv2D/ste_sign_3/add:z:0*
T0*&
_output_shapes
:882)
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1À
)quant_conv2d_2/Conv2D/ste_sign_3/IdentityIdentity+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0*
T0*&
_output_shapes
:882+
)quant_conv2d_2/Conv2D/ste_sign_3/Identityµ
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityN	IdentityN+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183607*8
_output_shapes&
$:88:882,
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityNý
quant_conv2d_2/Conv2DConv2D,quant_conv2d_2/ste_sign_4/IdentityN:output:03quant_conv2d_2/Conv2D/ste_sign_3/IdentityN:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*
paddingSAME*
strides
2
quant_conv2d_2/Conv2D
"batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/x
"batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/yÄ
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAnd¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:8*
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:8*
dtype02(
&batch_normalization_2/ReadVariableOp_1}
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_2/Const
batch_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_2/Const_1¡
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3quant_conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0$batch_normalization_2/Const:output:0&batch_normalization_2/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ 8:8:8:8:8:*
epsilon%o:2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_2/Const_2à
+batch_normalization_2/AssignMovingAvg/sub/xConst*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/183632*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_2/AssignMovingAvg/sub/x
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/183632*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/subÕ
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_2_assignmovingavg_183632*
_output_shapes
:8*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpº
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/183632*
_output_shapes
:82-
+batch_normalization_2/AssignMovingAvg/sub_1£
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/183632*
_output_shapes
:82+
)batch_normalization_2/AssignMovingAvg/mul
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_2_assignmovingavg_183632-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/183632*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpæ
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/183639*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_2/AssignMovingAvg_1/sub/x¥
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/183639*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/subÛ
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_2_assignmovingavg_1_183639*
_output_shapes
:8*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpÆ
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/183639*
_output_shapes
:82/
-batch_normalization_2/AssignMovingAvg_1/sub_1­
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/183639*
_output_shapes
:82-
+batch_normalization_2/AssignMovingAvg_1/mul
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_2_assignmovingavg_1_183639/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/183639*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpÖ
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape
quant_dense/ste_sign_6/SignSignflatten/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense/ste_sign_6/Sign
quant_dense/ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
quant_dense/ste_sign_6/add/y¼
quant_dense/ste_sign_6/addAddV2quant_dense/ste_sign_6/Sign:y:0%quant_dense/ste_sign_6/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense/ste_sign_6/add
quant_dense/ste_sign_6/Sign_1Signquant_dense/ste_sign_6/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense/ste_sign_6/Sign_1¤
quant_dense/ste_sign_6/IdentityIdentity!quant_dense/ste_sign_6/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
quant_dense/ste_sign_6/Identity
 quant_dense/ste_sign_6/IdentityN	IdentityN!quant_dense/ste_sign_6/Sign_1:y:0flatten/Reshape:output:0*
T
2*,
_gradient_op_typeCustomGradient-183649*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2"
 quant_dense/ste_sign_6/IdentityN³
!quant_dense/MatMul/ReadVariableOpReadVariableOp*quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!quant_dense/MatMul/ReadVariableOp¦
"quant_dense/MatMul/ste_sign_5/SignSign)quant_dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"quant_dense/MatMul/ste_sign_5/Sign
#quant_dense/MatMul/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2%
#quant_dense/MatMul/ste_sign_5/add/yÐ
!quant_dense/MatMul/ste_sign_5/addAddV2&quant_dense/MatMul/ste_sign_5/Sign:y:0,quant_dense/MatMul/ste_sign_5/add/y:output:0*
T0* 
_output_shapes
:
2#
!quant_dense/MatMul/ste_sign_5/add¦
$quant_dense/MatMul/ste_sign_5/Sign_1Sign%quant_dense/MatMul/ste_sign_5/add:z:0*
T0* 
_output_shapes
:
2&
$quant_dense/MatMul/ste_sign_5/Sign_1±
&quant_dense/MatMul/ste_sign_5/IdentityIdentity(quant_dense/MatMul/ste_sign_5/Sign_1:y:0*
T0* 
_output_shapes
:
2(
&quant_dense/MatMul/ste_sign_5/Identity
'quant_dense/MatMul/ste_sign_5/IdentityN	IdentityN(quant_dense/MatMul/ste_sign_5/Sign_1:y:0)quant_dense/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183659*,
_output_shapes
:
:
2)
'quant_dense/MatMul/ste_sign_5/IdentityNÂ
quant_dense/MatMulMatMul)quant_dense/ste_sign_6/IdentityN:output:00quant_dense/MatMul/ste_sign_5/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense/MatMul
"batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/x
"batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/yÄ
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAnd¶
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesè
"batch_normalization_3/moments/meanMeanquant_dense/MatMul:product:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2$
"batch_normalization_3/moments/mean¿
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:	2,
*batch_normalization_3/moments/StopGradientý
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencequant_dense/MatMul:product:03batch_normalization_3/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_3/moments/SquaredDifference¾
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indices
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2(
&batch_normalization_3/moments/varianceÃ
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2'
%batch_normalization_3/moments/SqueezeË
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1à
+batch_normalization_3/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/183679*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+batch_normalization_3/AssignMovingAvg/decayÖ
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_3_assignmovingavg_183679*
_output_shapes	
:*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp²
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/183679*
_output_shapes	
:2+
)batch_normalization_3/AssignMovingAvg/sub©
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/183679*
_output_shapes	
:2+
)batch_normalization_3/AssignMovingAvg/mul
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_3_assignmovingavg_183679-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/183679*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpæ
-batch_normalization_3/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/183685*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-batch_normalization_3/AssignMovingAvg_1/decayÜ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_3_assignmovingavg_1_183685*
_output_shapes	
:*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp¼
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/183685*
_output_shapes	
:2-
+batch_normalization_3/AssignMovingAvg_1/sub³
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/183685*
_output_shapes	
:2-
+batch_normalization_3/AssignMovingAvg_1/mul
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_3_assignmovingavg_1_183685/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/183685*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_3/batchnorm/add/yÛ
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/add¦
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_3/batchnorm/Rsqrtá
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/mulÏ
%batch_normalization_3/batchnorm/mul_1Mulquant_dense/MatMul:product:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/mul_1Ô
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_3/batchnorm/mul_2Õ
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpÚ
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/subÞ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/add_1¤
quant_dense_1/ste_sign_8/SignSign)batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense_1/ste_sign_8/Sign
quant_dense_1/ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2 
quant_dense_1/ste_sign_8/add/yÄ
quant_dense_1/ste_sign_8/addAddV2!quant_dense_1/ste_sign_8/Sign:y:0'quant_dense_1/ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense_1/ste_sign_8/add
quant_dense_1/ste_sign_8/Sign_1Sign quant_dense_1/ste_sign_8/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
quant_dense_1/ste_sign_8/Sign_1ª
!quant_dense_1/ste_sign_8/IdentityIdentity#quant_dense_1/ste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!quant_dense_1/ste_sign_8/Identity
"quant_dense_1/ste_sign_8/IdentityN	IdentityN#quant_dense_1/ste_sign_8/Sign_1:y:0)batch_normalization_3/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-183703*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2$
"quant_dense_1/ste_sign_8/IdentityN¹
#quant_dense_1/MatMul/ReadVariableOpReadVariableOp,quant_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#quant_dense_1/MatMul/ReadVariableOp¬
$quant_dense_1/MatMul/ste_sign_7/SignSign+quant_dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$quant_dense_1/MatMul/ste_sign_7/Sign
%quant_dense_1/MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2'
%quant_dense_1/MatMul/ste_sign_7/add/yØ
#quant_dense_1/MatMul/ste_sign_7/addAddV2(quant_dense_1/MatMul/ste_sign_7/Sign:y:0.quant_dense_1/MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
2%
#quant_dense_1/MatMul/ste_sign_7/add¬
&quant_dense_1/MatMul/ste_sign_7/Sign_1Sign'quant_dense_1/MatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
2(
&quant_dense_1/MatMul/ste_sign_7/Sign_1·
(quant_dense_1/MatMul/ste_sign_7/IdentityIdentity*quant_dense_1/MatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
2*
(quant_dense_1/MatMul/ste_sign_7/Identity¥
)quant_dense_1/MatMul/ste_sign_7/IdentityN	IdentityN*quant_dense_1/MatMul/ste_sign_7/Sign_1:y:0+quant_dense_1/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183713*,
_output_shapes
:
:
2+
)quant_dense_1/MatMul/ste_sign_7/IdentityNÊ
quant_dense_1/MatMulMatMul+quant_dense_1/ste_sign_8/IdentityN:output:02quant_dense_1/MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense_1/MatMul
"batch_normalization_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_4/LogicalAnd/x
"batch_normalization_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_4/LogicalAnd/yÄ
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_4/LogicalAnd¶
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesê
"batch_normalization_4/moments/meanMeanquant_dense_1/MatMul:product:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2$
"batch_normalization_4/moments/mean¿
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	2,
*batch_normalization_4/moments/StopGradientÿ
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencequant_dense_1/MatMul:product:03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_4/moments/SquaredDifference¾
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indices
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2(
&batch_normalization_4/moments/varianceÃ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2'
%batch_normalization_4/moments/SqueezeË
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1à
+batch_normalization_4/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/183733*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+batch_normalization_4/AssignMovingAvg/decayÖ
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_4_assignmovingavg_183733*
_output_shapes	
:*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp²
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/183733*
_output_shapes	
:2+
)batch_normalization_4/AssignMovingAvg/sub©
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/183733*
_output_shapes	
:2+
)batch_normalization_4/AssignMovingAvg/mul
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_4_assignmovingavg_183733-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/183733*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpæ
-batch_normalization_4/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/183739*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-batch_normalization_4/AssignMovingAvg_1/decayÜ
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_4_assignmovingavg_1_183739*
_output_shapes	
:*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp¼
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/183739*
_output_shapes	
:2-
+batch_normalization_4/AssignMovingAvg_1/sub³
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/183739*
_output_shapes	
:2-
+batch_normalization_4/AssignMovingAvg_1/mul
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_4_assignmovingavg_1_183739/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/183739*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yÛ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/add¦
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_4/batchnorm/Rsqrtá
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/mulÑ
%batch_normalization_4/batchnorm/mul_1Mulquant_dense_1/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_4/batchnorm/mul_1Ô
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_4/batchnorm/mul_2Õ
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpÚ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/subÞ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_4/batchnorm/add_1¦
quant_dense_2/ste_sign_10/SignSign)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
quant_dense_2/ste_sign_10/Sign
quant_dense_2/ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2!
quant_dense_2/ste_sign_10/add/yÈ
quant_dense_2/ste_sign_10/addAddV2"quant_dense_2/ste_sign_10/Sign:y:0(quant_dense_2/ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense_2/ste_sign_10/add¢
 quant_dense_2/ste_sign_10/Sign_1Sign!quant_dense_2/ste_sign_10/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 quant_dense_2/ste_sign_10/Sign_1­
"quant_dense_2/ste_sign_10/IdentityIdentity$quant_dense_2/ste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"quant_dense_2/ste_sign_10/Identity¡
#quant_dense_2/ste_sign_10/IdentityN	IdentityN$quant_dense_2/ste_sign_10/Sign_1:y:0)batch_normalization_4/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-183757*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2%
#quant_dense_2/ste_sign_10/IdentityN¸
#quant_dense_2/MatMul/ReadVariableOpReadVariableOp,quant_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#quant_dense_2/MatMul/ReadVariableOp«
$quant_dense_2/MatMul/ste_sign_9/SignSign+quant_dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$quant_dense_2/MatMul/ste_sign_9/Sign
%quant_dense_2/MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2'
%quant_dense_2/MatMul/ste_sign_9/add/y×
#quant_dense_2/MatMul/ste_sign_9/addAddV2(quant_dense_2/MatMul/ste_sign_9/Sign:y:0.quant_dense_2/MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	2%
#quant_dense_2/MatMul/ste_sign_9/add«
&quant_dense_2/MatMul/ste_sign_9/Sign_1Sign'quant_dense_2/MatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	2(
&quant_dense_2/MatMul/ste_sign_9/Sign_1¶
(quant_dense_2/MatMul/ste_sign_9/IdentityIdentity*quant_dense_2/MatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	2*
(quant_dense_2/MatMul/ste_sign_9/Identity£
)quant_dense_2/MatMul/ste_sign_9/IdentityN	IdentityN*quant_dense_2/MatMul/ste_sign_9/Sign_1:y:0+quant_dense_2/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183767**
_output_shapes
:	:	2+
)quant_dense_2/MatMul/ste_sign_9/IdentityNÊ
quant_dense_2/MatMulMatMul,quant_dense_2/ste_sign_10/IdentityN:output:02quant_dense_2/MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense_2/MatMul
"batch_normalization_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_5/LogicalAnd/x
"batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_5/LogicalAnd/yÄ
 batch_normalization_5/LogicalAnd
LogicalAnd+batch_normalization_5/LogicalAnd/x:output:0+batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_5/LogicalAnd¶
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_5/moments/mean/reduction_indicesé
"batch_normalization_5/moments/meanMeanquant_dense_2/MatMul:product:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_5/moments/mean¾
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_5/moments/StopGradientþ
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencequant_dense_2/MatMul:product:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_5/moments/SquaredDifference¾
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_5/moments/variance/reduction_indices
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_5/moments/varianceÂ
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_5/moments/SqueezeÊ
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1à
+batch_normalization_5/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/183787*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+batch_normalization_5/AssignMovingAvg/decayÕ
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_5_assignmovingavg_183787*
_output_shapes
:*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOp±
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/183787*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/sub¨
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/183787*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/mul
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_5_assignmovingavg_183787-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/183787*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpæ
-batch_normalization_5/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/183793*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-batch_normalization_5/AssignMovingAvg_1/decayÛ
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_5_assignmovingavg_1_183793*
_output_shapes
:*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp»
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/183793*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/sub²
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/183793*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/mul
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_5_assignmovingavg_1_183793/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/183793*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_5/batchnorm/add/yÚ
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/add¥
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtà
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mulÐ
%batch_normalization_5/batchnorm/mul_1Mulquant_dense_2/MatMul:product:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_5/batchnorm/mul_1Ó
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2Ô
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpÙ
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/subÝ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_5/batchnorm/add_1
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Softmaxå
IdentityIdentityactivation/Softmax:softmax:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp#^quant_conv2d/Conv2D/ReadVariableOp%^quant_conv2d_1/Conv2D/ReadVariableOp%^quant_conv2d_2/Conv2D/ReadVariableOp"^quant_dense/MatMul/ReadVariableOp$^quant_dense_1/MatMul/ReadVariableOp$^quant_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2H
"quant_conv2d/Conv2D/ReadVariableOp"quant_conv2d/Conv2D/ReadVariableOp2L
$quant_conv2d_1/Conv2D/ReadVariableOp$quant_conv2d_1/Conv2D/ReadVariableOp2L
$quant_conv2d_2/Conv2D/ReadVariableOp$quant_conv2d_2/Conv2D/ReadVariableOp2F
!quant_dense/MatMul/ReadVariableOp!quant_dense/MatMul/ReadVariableOp2J
#quant_dense_1/MatMul/ReadVariableOp#quant_dense_1/MatMul/ReadVariableOp2J
#quant_dense_2/MatMul/ReadVariableOp#quant_dense_2/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
­%

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184650

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_184635
assignmovingavg_1_184642
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ 8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/184635*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/184635*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_184635*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/184635*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/184635*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_184635AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/184635*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/184642*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184642*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_184642*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184642*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184642*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_184642AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/184642*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¦
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ 8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

¤
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_182778

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOpe
ste_sign_8/SignSigninputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_8/Signi
ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
ste_sign_8/add/y
ste_sign_8/addAddV2ste_sign_8/Sign:y:0ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_8/addu
ste_sign_8/Sign_1Signste_sign_8/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_8/Sign_1
ste_sign_8/IdentityIdentityste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_8/IdentityÑ
ste_sign_8/IdentityN	IdentityNste_sign_8/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-182758*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
ste_sign_8/IdentityN
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOp
MatMul/ste_sign_7/SignSignMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_7/Signw
MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
MatMul/ste_sign_7/add/y 
MatMul/ste_sign_7/addAddV2MatMul/ste_sign_7/Sign:y:0 MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_7/add
MatMul/ste_sign_7/Sign_1SignMatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_7/Sign_1
MatMul/ste_sign_7/IdentityIdentityMatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_7/Identityí
MatMul/ste_sign_7/IdentityN	IdentityNMatMul/ste_sign_7/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-182768*,
_output_shapes
:
:
2
MatMul/ste_sign_7/IdentityN
MatMulMatMulste_sign_8/IdentityN:output:0$MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ã
Ã
+__inference_sequential_layer_call_fn_183283
quant_conv2d_input
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

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1832202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namequant_conv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
·
¼
$__inference_signature_wrapper_183502
quant_conv2d_input
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

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_1813172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namequant_conv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ñ
d
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_185210

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:882
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:882
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:882
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:882

Identity¬
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-185201*8
_output_shapes&
$:88:882
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:882

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:88:N J
&
_output_shapes
:88
 
_user_specified_nameinputs
Ñ
d
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_185244

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:882
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:882
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:882
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:882

Identity¬
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-185235*8
_output_shapes&
$:88:882
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:882

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:88:N J
&
_output_shapes
:88
 
_user_specified_nameinputs
º
ò
O__inference_batch_normalization_layer_call_and_return_conditional_losses_181483

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¿
·
+__inference_sequential_layer_call_fn_184170

inputs
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

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1832202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Õ0
È
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_182031

inputs
assignmovingavg_182006
assignmovingavg_1_182012)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/182006*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_182006*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpÄ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/182006*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/182006*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_182006AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/182006*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/182012*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_182012*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÎ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182012*
_output_shapes	
:2
AssignMovingAvg_1/subÅ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182012*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_182012AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/182012*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1´
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Í
E
)__inference_ste_sign_layer_call_fn_185198

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*&
_output_shapes
:8*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_ste_sign_layer_call_and_return_conditional_losses_1813352
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:82

Identity"
identityIdentity:output:0*%
_input_shapes
:8:N J
&
_output_shapes
:8
 
_user_specified_nameinputs
Õ0
È
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184968

inputs
assignmovingavg_184943
assignmovingavg_1_184949)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/184943*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_184943*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpÄ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/184943*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/184943*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_184943AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/184943*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/184949*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_184949*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÎ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184949*
_output_shapes	
:2
AssignMovingAvg_1/subÅ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184949*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_184949AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/184949*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1´
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
õ%

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_181868

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_181853
assignmovingavg_1_181860
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/181853*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/181853*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_181853*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/181853*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/181853*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_181853AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/181853*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/181860*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/181860*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_181860*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/181860*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/181860*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_181860AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/181860*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¸
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ì
L
0__inference_max_pooling2d_1_layer_call_fn_181716

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1817102
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­%

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182612

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_182597
assignmovingavg_1_182604
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ 8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/182597*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/182597*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_182597*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/182597*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/182597*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_182597AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/182597*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/182604*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182604*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_182604*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182604*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182604*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_182604AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/182604*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¦
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ 8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ú

Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_182371

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
öY
Á
F__inference_sequential_layer_call_and_return_conditional_losses_182909
quant_conv2d_input
quant_conv2d_182386
batch_normalization_182471
batch_normalization_182473
batch_normalization_182475
batch_normalization_182477
quant_conv2d_1_182481 
batch_normalization_1_182566 
batch_normalization_1_182568 
batch_normalization_1_182570 
batch_normalization_1_182572
quant_conv2d_2_182576 
batch_normalization_2_182661 
batch_normalization_2_182663 
batch_normalization_2_182665 
batch_normalization_2_182667
quant_dense_182717 
batch_normalization_3_182746 
batch_normalization_3_182748 
batch_normalization_3_182750 
batch_normalization_3_182752
quant_dense_1_182787 
batch_normalization_4_182816 
batch_normalization_4_182818 
batch_normalization_4_182820 
batch_normalization_4_182822
quant_dense_2_182857 
batch_normalization_5_182886 
batch_normalization_5_182888 
batch_normalization_5_182890 
batch_normalization_5_182892
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢#quant_dense/StatefulPartitionedCall¢%quant_dense_1/StatefulPartitionedCall¢%quant_dense_2/StatefulPartitionedCallâ
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputquant_conv2d_182386*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1813462&
$quant_conv2d/StatefulPartitionedCallð
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_182471batch_normalization_182473batch_normalization_182475batch_normalization_182477*
Tin	
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1824222-
+batch_normalization/StatefulPartitionedCallØ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1815002
max_pooling2d/PartitionedCallý
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_182481*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1815562(
&quant_conv2d_1/StatefulPartitionedCallÿ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_182566batch_normalization_1_182568batch_normalization_1_182570batch_normalization_1_182572*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825172/
-batch_normalization_1/StatefulPartitionedCallà
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1817102!
max_pooling2d_1/PartitionedCallÿ
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_182576*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1817662(
&quant_conv2d_2/StatefulPartitionedCallÿ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_182661batch_normalization_2_182663batch_normalization_2_182665batch_normalization_2_182667*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1826122/
-batch_normalization_2/StatefulPartitionedCallà
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1819202!
max_pooling2d_2/PartitionedCall³
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1826772
flatten/PartitionedCallä
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_182717*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_1827082%
#quant_dense/StatefulPartitionedCallõ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_182746batch_normalization_3_182748batch_normalization_3_182750batch_normalization_3_182752*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1820312/
-batch_normalization_3/StatefulPartitionedCall
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_182787*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1827782'
%quant_dense_1/StatefulPartitionedCall÷
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_182816batch_normalization_4_182818batch_normalization_4_182820batch_normalization_4_182822*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1821832/
-batch_normalization_4/StatefulPartitionedCall
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_182857*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1828482'
%quant_dense_2/StatefulPartitionedCallö
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_182886batch_normalization_5_182888batch_normalization_5_182890batch_normalization_5_182892*
Tin	
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1823352/
-batch_normalization_5/StatefulPartitionedCallÉ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1829002
activation/PartitionedCall
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall:d `
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namequant_conv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

§
4__inference_batch_normalization_layer_call_fn_184346

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1824442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ8::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ì
©
6__inference_batch_normalization_4_layer_call_fn_185017

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1822192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

¥
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_181766

inputs"
conv2d_readvariableop_resource
identity¢Conv2D/ReadVariableOp³
ste_sign_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_1817322
ste_sign_4/PartitionedCall§
ste_sign_4/IdentityIdentity#ste_sign_4/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
ste_sign_4/Identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02
Conv2D/ReadVariableOp½
!Conv2D/ste_sign_3/PartitionedCallPartitionedCallConv2D/ReadVariableOp:value:0*
Tin
2*
Tout
2*&
_output_shapes
:88*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_1817552#
!Conv2D/ste_sign_3/PartitionedCall¡
Conv2D/ste_sign_3/IdentityIdentity*Conv2D/ste_sign_3/PartitionedCall:output:0*
T0*&
_output_shapes
:882
Conv2D/ste_sign_3/IdentityÑ
Conv2DConv2Dste_sign_4/Identity:output:0#Conv2D/ste_sign_3/Identity:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*
paddingSAME*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: 
Ñ
d
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_181755

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:882
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:882
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:882
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:882

Identity¬
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-181746*8
_output_shapes&
$:88:882
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:882

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:88:N J
&
_output_shapes
:88
 
_user_specified_nameinputs
¼
ô
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184590

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Å	
d
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_181732

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
add/yy
addAddV2Sign:y:0add/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identityâ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-181723*n
_output_shapes\
Z:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
	IdentityN

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
º
ò
O__inference_batch_normalization_layer_call_and_return_conditional_losses_184238

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
½0
È
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_182335

inputs
assignmovingavg_182310
assignmovingavg_1_182316)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/182310*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_182310*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpÃ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/182310*
_output_shapes
:2
AssignMovingAvg/subº
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/182310*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_182310AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/182310*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/182316*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_182316*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÍ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182316*
_output_shapes
:2
AssignMovingAvg_1/subÄ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182316*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_182316AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/182316*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¿
·
+__inference_sequential_layer_call_fn_184105

inputs
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

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1830742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ô
ô
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182539

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿA
8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstÚ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿA
8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ì
§
4__inference_batch_normalization_layer_call_fn_184264

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1814832
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

s
-__inference_quant_conv2d_layer_call_fn_181354

inputs
unknown
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1813462
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Ð
©
6__inference_batch_normalization_1_layer_call_fn_184427

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1816582
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ñ
G
+__inference_ste_sign_1_layer_call_fn_185215

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*&
_output_shapes
:88*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_1815452
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:882

Identity"
identityIdentity:output:0*%
_input_shapes
:88:N J
&
_output_shapes
:88
 
_user_specified_nameinputs

¢
G__inference_quant_dense_layer_call_and_return_conditional_losses_182708

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOpe
ste_sign_6/SignSigninputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_6/Signi
ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
ste_sign_6/add/y
ste_sign_6/addAddV2ste_sign_6/Sign:y:0ste_sign_6/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_6/addu
ste_sign_6/Sign_1Signste_sign_6/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_6/Sign_1
ste_sign_6/IdentityIdentityste_sign_6/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_6/IdentityÑ
ste_sign_6/IdentityN	IdentityNste_sign_6/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-182688*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
ste_sign_6/IdentityN
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOp
MatMul/ste_sign_5/SignSignMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_5/Signw
MatMul/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
MatMul/ste_sign_5/add/y 
MatMul/ste_sign_5/addAddV2MatMul/ste_sign_5/Sign:y:0 MatMul/ste_sign_5/add/y:output:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_5/add
MatMul/ste_sign_5/Sign_1SignMatMul/ste_sign_5/add:z:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_5/Sign_1
MatMul/ste_sign_5/IdentityIdentityMatMul/ste_sign_5/Sign_1:y:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_5/Identityí
MatMul/ste_sign_5/IdentityN	IdentityNMatMul/ste_sign_5/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-182698*,
_output_shapes
:
:
2
MatMul/ste_sign_5/IdentityN
MatMulMatMulste_sign_6/IdentityN:output:0$MatMul/ste_sign_5/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 

¤
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_182848

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOpg
ste_sign_10/SignSigninputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_10/Signk
ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
ste_sign_10/add/y
ste_sign_10/addAddV2ste_sign_10/Sign:y:0ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_10/addx
ste_sign_10/Sign_1Signste_sign_10/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_10/Sign_1
ste_sign_10/IdentityIdentityste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_10/IdentityÔ
ste_sign_10/IdentityN	IdentityNste_sign_10/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-182828*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
ste_sign_10/IdentityN
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOp
MatMul/ste_sign_9/SignSignMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul/ste_sign_9/Signw
MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
MatMul/ste_sign_9/add/y
MatMul/ste_sign_9/addAddV2MatMul/ste_sign_9/Sign:y:0 MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	2
MatMul/ste_sign_9/add
MatMul/ste_sign_9/Sign_1SignMatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	2
MatMul/ste_sign_9/Sign_1
MatMul/ste_sign_9/IdentityIdentityMatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	2
MatMul/ste_sign_9/Identityë
MatMul/ste_sign_9/IdentityN	IdentityNMatMul/ste_sign_9/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-182838**
_output_shapes
:	:	2
MatMul/ste_sign_9/IdentityN
MatMulMatMulste_sign_10/IdentityN:output:0$MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ì
©
6__inference_batch_normalization_3_layer_call_fn_184850

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1820312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

¤
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_185040

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOpg
ste_sign_10/SignSigninputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_10/Signk
ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
ste_sign_10/add/y
ste_sign_10/addAddV2ste_sign_10/Sign:y:0ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_10/addx
ste_sign_10/Sign_1Signste_sign_10/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_10/Sign_1
ste_sign_10/IdentityIdentityste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_10/IdentityÔ
ste_sign_10/IdentityN	IdentityNste_sign_10/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-185020*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
ste_sign_10/IdentityN
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOp
MatMul/ste_sign_9/SignSignMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul/ste_sign_9/Signw
MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
MatMul/ste_sign_9/add/y
MatMul/ste_sign_9/addAddV2MatMul/ste_sign_9/Sign:y:0 MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	2
MatMul/ste_sign_9/add
MatMul/ste_sign_9/Sign_1SignMatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	2
MatMul/ste_sign_9/Sign_1
MatMul/ste_sign_9/IdentityIdentityMatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	2
MatMul/ste_sign_9/Identityë
MatMul/ste_sign_9/IdentityN	IdentityNMatMul/ste_sign_9/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-185030**
_output_shapes
:	:	2
MatMul/ste_sign_9/IdentityN
MatMulMatMulste_sign_10/IdentityN:output:0$MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
è
J
.__inference_max_pooling2d_layer_call_fn_181506

inputs
identity©
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1815002
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
ô
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_184496

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿA
8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstÚ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿA
8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó%

O__inference_batch_normalization_layer_call_and_return_conditional_losses_181448

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_181433
assignmovingavg_1_181440
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/181433*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/181433*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_181433*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/181433*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/181433*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_181433AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/181433*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/181440*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/181440*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_181440*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/181440*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/181440*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_181440AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/181440*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¸
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ÒY
µ
F__inference_sequential_layer_call_and_return_conditional_losses_183074

inputs
quant_conv2d_182996
batch_normalization_182999
batch_normalization_183001
batch_normalization_183003
batch_normalization_183005
quant_conv2d_1_183009 
batch_normalization_1_183012 
batch_normalization_1_183014 
batch_normalization_1_183016 
batch_normalization_1_183018
quant_conv2d_2_183022 
batch_normalization_2_183025 
batch_normalization_2_183027 
batch_normalization_2_183029 
batch_normalization_2_183031
quant_dense_183036 
batch_normalization_3_183039 
batch_normalization_3_183041 
batch_normalization_3_183043 
batch_normalization_3_183045
quant_dense_1_183048 
batch_normalization_4_183051 
batch_normalization_4_183053 
batch_normalization_4_183055 
batch_normalization_4_183057
quant_dense_2_183060 
batch_normalization_5_183063 
batch_normalization_5_183065 
batch_normalization_5_183067 
batch_normalization_5_183069
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢#quant_dense/StatefulPartitionedCall¢%quant_dense_1/StatefulPartitionedCall¢%quant_dense_2/StatefulPartitionedCallÖ
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsquant_conv2d_182996*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1813462&
$quant_conv2d/StatefulPartitionedCallð
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_182999batch_normalization_183001batch_normalization_183003batch_normalization_183005*
Tin	
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1824222-
+batch_normalization/StatefulPartitionedCallØ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1815002
max_pooling2d/PartitionedCallý
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_183009*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1815562(
&quant_conv2d_1/StatefulPartitionedCallÿ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_183012batch_normalization_1_183014batch_normalization_1_183016batch_normalization_1_183018*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825172/
-batch_normalization_1/StatefulPartitionedCallà
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1817102!
max_pooling2d_1/PartitionedCallÿ
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_183022*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1817662(
&quant_conv2d_2/StatefulPartitionedCallÿ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_183025batch_normalization_2_183027batch_normalization_2_183029batch_normalization_2_183031*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1826122/
-batch_normalization_2/StatefulPartitionedCallà
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1819202!
max_pooling2d_2/PartitionedCall³
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1826772
flatten/PartitionedCallä
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_183036*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_1827082%
#quant_dense/StatefulPartitionedCallõ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_183039batch_normalization_3_183041batch_normalization_3_183043batch_normalization_3_183045*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1820312/
-batch_normalization_3/StatefulPartitionedCall
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_183048*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1827782'
%quant_dense_1/StatefulPartitionedCall÷
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_183051batch_normalization_4_183053batch_normalization_4_183055batch_normalization_4_183057*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1821832/
-batch_normalization_4/StatefulPartitionedCall
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_183060*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1828482'
%quant_dense_2/StatefulPartitionedCallö
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_183063batch_normalization_5_183065batch_normalization_5_183067batch_normalization_5_183069*
Tin	
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1823352/
-batch_normalization_5/StatefulPartitionedCallÉ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1829002
activation/PartitionedCall
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
öY
Á
F__inference_sequential_layer_call_and_return_conditional_losses_182990
quant_conv2d_input
quant_conv2d_182912
batch_normalization_182915
batch_normalization_182917
batch_normalization_182919
batch_normalization_182921
quant_conv2d_1_182925 
batch_normalization_1_182928 
batch_normalization_1_182930 
batch_normalization_1_182932 
batch_normalization_1_182934
quant_conv2d_2_182938 
batch_normalization_2_182941 
batch_normalization_2_182943 
batch_normalization_2_182945 
batch_normalization_2_182947
quant_dense_182952 
batch_normalization_3_182955 
batch_normalization_3_182957 
batch_normalization_3_182959 
batch_normalization_3_182961
quant_dense_1_182964 
batch_normalization_4_182967 
batch_normalization_4_182969 
batch_normalization_4_182971 
batch_normalization_4_182973
quant_dense_2_182976 
batch_normalization_5_182979 
batch_normalization_5_182981 
batch_normalization_5_182983 
batch_normalization_5_182985
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢#quant_dense/StatefulPartitionedCall¢%quant_dense_1/StatefulPartitionedCall¢%quant_dense_2/StatefulPartitionedCallâ
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputquant_conv2d_182912*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1813462&
$quant_conv2d/StatefulPartitionedCallð
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_182915batch_normalization_182917batch_normalization_182919batch_normalization_182921*
Tin	
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1824442-
+batch_normalization/StatefulPartitionedCallØ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1815002
max_pooling2d/PartitionedCallý
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_182925*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1815562(
&quant_conv2d_1/StatefulPartitionedCallÿ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_182928batch_normalization_1_182930batch_normalization_1_182932batch_normalization_1_182934*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825392/
-batch_normalization_1/StatefulPartitionedCallà
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1817102!
max_pooling2d_1/PartitionedCallÿ
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_182938*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1817662(
&quant_conv2d_2/StatefulPartitionedCallÿ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_182941batch_normalization_2_182943batch_normalization_2_182945batch_normalization_2_182947*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1826342/
-batch_normalization_2/StatefulPartitionedCallà
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1819202!
max_pooling2d_2/PartitionedCall³
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1826772
flatten/PartitionedCallä
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_182952*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_1827082%
#quant_dense/StatefulPartitionedCallõ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_182955batch_normalization_3_182957batch_normalization_3_182959batch_normalization_3_182961*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1820672/
-batch_normalization_3/StatefulPartitionedCall
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_182964*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1827782'
%quant_dense_1/StatefulPartitionedCall÷
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_182967batch_normalization_4_182969batch_normalization_4_182971batch_normalization_4_182973*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1822192/
-batch_normalization_4/StatefulPartitionedCall
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_182976*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1828482'
%quant_dense_2/StatefulPartitionedCallö
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_182979batch_normalization_5_182981batch_normalization_5_182983batch_normalization_5_182985*
Tin	
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1823712/
-batch_normalization_5/StatefulPartitionedCallÉ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1829002
activation/PartitionedCall
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall:d `
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namequant_conv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ö
ò
O__inference_batch_normalization_layer_call_and_return_conditional_losses_182444

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ë
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstÛ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
­%

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_184474

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_184459
assignmovingavg_1_184466
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿA
8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/184459*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/184459*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_184459*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/184459*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/184459*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_184459AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/184459*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/184466*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184466*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_184466*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184466*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184466*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_184466AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/184466*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¦
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿA
8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_182219

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_181710

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½0
È
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_185122

inputs
assignmovingavg_185097
assignmovingavg_1_185103)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/185097*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_185097*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpÃ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/185097*
_output_shapes
:2
AssignMovingAvg/subº
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/185097*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_185097AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/185097*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/185103*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_185103*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÍ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/185103*
_output_shapes
:2
AssignMovingAvg_1/subÄ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/185103*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_185103AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/185103*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
õ%

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_184392

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_184377
assignmovingavg_1_184384
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/184377*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/184377*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_184377*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/184377*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/184377*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_184377AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/184377*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/184384*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184384*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_184384*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184384*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184384*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_184384AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/184384*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¸
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ð
©
6__inference_batch_normalization_2_layer_call_fn_184603

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1818682
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
­%

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_182517

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_182502
assignmovingavg_1_182509
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿA
8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/182502*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/182502*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_182502*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/182502*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/182502*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_182502AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/182502*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/182509*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182509*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_182509*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182509*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182509*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_182509AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/182509*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¦
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿA
8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

©
6__inference_batch_normalization_2_layer_call_fn_184698

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1826342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ 8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_184837

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ÿ
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_181500

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
©
6__inference_batch_normalization_1_layer_call_fn_184440

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1816932
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ð
©
6__inference_batch_normalization_2_layer_call_fn_184616

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1819032
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184991

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
îñ
Ò
!__inference__wrapped_model_181317
quant_conv2d_input:
6sequential_quant_conv2d_conv2d_readvariableop_resource:
6sequential_batch_normalization_readvariableop_resource<
8sequential_batch_normalization_readvariableop_1_resourceK
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource<
8sequential_quant_conv2d_1_conv2d_readvariableop_resource<
8sequential_batch_normalization_1_readvariableop_resource>
:sequential_batch_normalization_1_readvariableop_1_resourceM
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource<
8sequential_quant_conv2d_2_conv2d_readvariableop_resource<
8sequential_batch_normalization_2_readvariableop_resource>
:sequential_batch_normalization_2_readvariableop_1_resourceM
Isequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource9
5sequential_quant_dense_matmul_readvariableop_resourceF
Bsequential_batch_normalization_3_batchnorm_readvariableop_resourceJ
Fsequential_batch_normalization_3_batchnorm_mul_readvariableop_resourceH
Dsequential_batch_normalization_3_batchnorm_readvariableop_1_resourceH
Dsequential_batch_normalization_3_batchnorm_readvariableop_2_resource;
7sequential_quant_dense_1_matmul_readvariableop_resourceF
Bsequential_batch_normalization_4_batchnorm_readvariableop_resourceJ
Fsequential_batch_normalization_4_batchnorm_mul_readvariableop_resourceH
Dsequential_batch_normalization_4_batchnorm_readvariableop_1_resourceH
Dsequential_batch_normalization_4_batchnorm_readvariableop_2_resource;
7sequential_quant_dense_2_matmul_readvariableop_resourceF
Bsequential_batch_normalization_5_batchnorm_readvariableop_resourceJ
Fsequential_batch_normalization_5_batchnorm_mul_readvariableop_resourceH
Dsequential_batch_normalization_5_batchnorm_readvariableop_1_resourceH
Dsequential_batch_normalization_5_batchnorm_readvariableop_2_resource
identity¢>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp¢@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢-sequential/batch_normalization/ReadVariableOp¢/sequential/batch_normalization/ReadVariableOp_1¢@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢/sequential/batch_normalization_1/ReadVariableOp¢1sequential/batch_normalization_1/ReadVariableOp_1¢@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢/sequential/batch_normalization_2/ReadVariableOp¢1sequential/batch_normalization_2/ReadVariableOp_1¢9sequential/batch_normalization_3/batchnorm/ReadVariableOp¢;sequential/batch_normalization_3/batchnorm/ReadVariableOp_1¢;sequential/batch_normalization_3/batchnorm/ReadVariableOp_2¢=sequential/batch_normalization_3/batchnorm/mul/ReadVariableOp¢9sequential/batch_normalization_4/batchnorm/ReadVariableOp¢;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1¢;sequential/batch_normalization_4/batchnorm/ReadVariableOp_2¢=sequential/batch_normalization_4/batchnorm/mul/ReadVariableOp¢9sequential/batch_normalization_5/batchnorm/ReadVariableOp¢;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1¢;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2¢=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOp¢-sequential/quant_conv2d/Conv2D/ReadVariableOp¢/sequential/quant_conv2d_1/Conv2D/ReadVariableOp¢/sequential/quant_conv2d_2/Conv2D/ReadVariableOp¢,sequential/quant_dense/MatMul/ReadVariableOp¢.sequential/quant_dense_1/MatMul/ReadVariableOp¢.sequential/quant_dense_2/MatMul/ReadVariableOpÝ
-sequential/quant_conv2d/Conv2D/ReadVariableOpReadVariableOp6sequential_quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02/
-sequential/quant_conv2d/Conv2D/ReadVariableOpÌ
,sequential/quant_conv2d/Conv2D/ste_sign/SignSign5sequential/quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:82.
,sequential/quant_conv2d/Conv2D/ste_sign/Sign£
-sequential/quant_conv2d/Conv2D/ste_sign/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-sequential/quant_conv2d/Conv2D/ste_sign/add/yþ
+sequential/quant_conv2d/Conv2D/ste_sign/addAddV20sequential/quant_conv2d/Conv2D/ste_sign/Sign:y:06sequential/quant_conv2d/Conv2D/ste_sign/add/y:output:0*
T0*&
_output_shapes
:82-
+sequential/quant_conv2d/Conv2D/ste_sign/addÊ
.sequential/quant_conv2d/Conv2D/ste_sign/Sign_1Sign/sequential/quant_conv2d/Conv2D/ste_sign/add:z:0*
T0*&
_output_shapes
:820
.sequential/quant_conv2d/Conv2D/ste_sign/Sign_1Õ
0sequential/quant_conv2d/Conv2D/ste_sign/IdentityIdentity2sequential/quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*
T0*&
_output_shapes
:822
0sequential/quant_conv2d/Conv2D/ste_sign/IdentityÓ
1sequential/quant_conv2d/Conv2D/ste_sign/IdentityN	IdentityN2sequential/quant_conv2d/Conv2D/ste_sign/Sign_1:y:05sequential/quant_conv2d/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-181095*8
_output_shapes&
$:8:823
1sequential/quant_conv2d/Conv2D/ste_sign/IdentityNý
sequential/quant_conv2d/Conv2DConv2Dquant_conv2d_input:sequential/quant_conv2d/Conv2D/ste_sign/IdentityN:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
paddingSAME*
strides
2 
sequential/quant_conv2d/Conv2D
+sequential/batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2-
+sequential/batch_normalization/LogicalAnd/x
+sequential/batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2-
+sequential/batch_normalization/LogicalAnd/yè
)sequential/batch_normalization/LogicalAnd
LogicalAnd4sequential/batch_normalization/LogicalAnd/x:output:04sequential/batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2+
)sequential/batch_normalization/LogicalAndÑ
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype02/
-sequential/batch_normalization/ReadVariableOp×
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization/ReadVariableOp_1
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¦
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3'sequential/quant_conv2d/Conv2D:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3
$sequential/batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2&
$sequential/batch_normalization/Constñ
 sequential/max_pooling2d/MaxPoolMaxPool3sequential/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolÃ
)sequential/quant_conv2d_1/ste_sign_2/SignSign)sequential/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82+
)sequential/quant_conv2d_1/ste_sign_2/Sign
*sequential/quant_conv2d_1/ste_sign_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*sequential/quant_conv2d_1/ste_sign_2/add/yû
(sequential/quant_conv2d_1/ste_sign_2/addAddV2-sequential/quant_conv2d_1/ste_sign_2/Sign:y:03sequential/quant_conv2d_1/ste_sign_2/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82*
(sequential/quant_conv2d_1/ste_sign_2/addÊ
+sequential/quant_conv2d_1/ste_sign_2/Sign_1Sign,sequential/quant_conv2d_1/ste_sign_2/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82-
+sequential/quant_conv2d_1/ste_sign_2/Sign_1Õ
-sequential/quant_conv2d_1/ste_sign_2/IdentityIdentity/sequential/quant_conv2d_1/ste_sign_2/Sign_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82/
-sequential/quant_conv2d_1/ste_sign_2/IdentityÐ
.sequential/quant_conv2d_1/ste_sign_2/IdentityN	IdentityN/sequential/quant_conv2d_1/ste_sign_2/Sign_1:y:0)sequential/max_pooling2d/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-181123*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿA
8:ÿÿÿÿÿÿÿÿÿA
820
.sequential/quant_conv2d_1/ste_sign_2/IdentityNã
/sequential/quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp8sequential_quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype021
/sequential/quant_conv2d_1/Conv2D/ReadVariableOpÖ
0sequential/quant_conv2d_1/Conv2D/ste_sign_1/SignSign7sequential/quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:8822
0sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign«
1sequential/quant_conv2d_1/Conv2D/ste_sign_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=23
1sequential/quant_conv2d_1/Conv2D/ste_sign_1/add/y
/sequential/quant_conv2d_1/Conv2D/ste_sign_1/addAddV24sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign:y:0:sequential/quant_conv2d_1/Conv2D/ste_sign_1/add/y:output:0*
T0*&
_output_shapes
:8821
/sequential/quant_conv2d_1/Conv2D/ste_sign_1/addÖ
2sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1Sign3sequential/quant_conv2d_1/Conv2D/ste_sign_1/add:z:0*
T0*&
_output_shapes
:8824
2sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1á
4sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityIdentity6sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0*
T0*&
_output_shapes
:8826
4sequential/quant_conv2d_1/Conv2D/ste_sign_1/Identityá
5sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityN	IdentityN6sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:07sequential/quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-181133*8
_output_shapes&
$:88:8827
5sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityN©
 sequential/quant_conv2d_1/Conv2DConv2D7sequential/quant_conv2d_1/ste_sign_2/IdentityN:output:0>sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityN:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*
paddingSAME*
strides
2"
 sequential/quant_conv2d_1/Conv2D 
-sequential/batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-sequential/batch_normalization_1/LogicalAnd/x 
-sequential/batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-sequential/batch_normalization_1/LogicalAnd/yð
+sequential/batch_normalization_1/LogicalAnd
LogicalAnd6sequential/batch_normalization_1/LogicalAnd/x:output:06sequential/batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_1/LogicalAnd×
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization_1/ReadVariableOpÝ
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:8*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1³
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3)sequential/quant_conv2d_1/Conv2D:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿA
8:8:8:8:8:*
epsilon%o:*
is_training( 23
1sequential/batch_normalization_1/FusedBatchNormV3
&sequential/batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2(
&sequential/batch_normalization_1/Const÷
"sequential/max_pooling2d_1/MaxPoolMaxPool5sequential/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPoolÅ
)sequential/quant_conv2d_2/ste_sign_4/SignSign+sequential/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82+
)sequential/quant_conv2d_2/ste_sign_4/Sign
*sequential/quant_conv2d_2/ste_sign_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*sequential/quant_conv2d_2/ste_sign_4/add/yû
(sequential/quant_conv2d_2/ste_sign_4/addAddV2-sequential/quant_conv2d_2/ste_sign_4/Sign:y:03sequential/quant_conv2d_2/ste_sign_4/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82*
(sequential/quant_conv2d_2/ste_sign_4/addÊ
+sequential/quant_conv2d_2/ste_sign_4/Sign_1Sign,sequential/quant_conv2d_2/ste_sign_4/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82-
+sequential/quant_conv2d_2/ste_sign_4/Sign_1Õ
-sequential/quant_conv2d_2/ste_sign_4/IdentityIdentity/sequential/quant_conv2d_2/ste_sign_4/Sign_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82/
-sequential/quant_conv2d_2/ste_sign_4/IdentityÒ
.sequential/quant_conv2d_2/ste_sign_4/IdentityN	IdentityN/sequential/quant_conv2d_2/ste_sign_4/Sign_1:y:0+sequential/max_pooling2d_1/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-181161*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ 8:ÿÿÿÿÿÿÿÿÿ 820
.sequential/quant_conv2d_2/ste_sign_4/IdentityNã
/sequential/quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp8sequential_quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype021
/sequential/quant_conv2d_2/Conv2D/ReadVariableOpÖ
0sequential/quant_conv2d_2/Conv2D/ste_sign_3/SignSign7sequential/quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:8822
0sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign«
1sequential/quant_conv2d_2/Conv2D/ste_sign_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=23
1sequential/quant_conv2d_2/Conv2D/ste_sign_3/add/y
/sequential/quant_conv2d_2/Conv2D/ste_sign_3/addAddV24sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign:y:0:sequential/quant_conv2d_2/Conv2D/ste_sign_3/add/y:output:0*
T0*&
_output_shapes
:8821
/sequential/quant_conv2d_2/Conv2D/ste_sign_3/addÖ
2sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1Sign3sequential/quant_conv2d_2/Conv2D/ste_sign_3/add:z:0*
T0*&
_output_shapes
:8824
2sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1á
4sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityIdentity6sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0*
T0*&
_output_shapes
:8826
4sequential/quant_conv2d_2/Conv2D/ste_sign_3/Identityá
5sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityN	IdentityN6sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:07sequential/quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-181171*8
_output_shapes&
$:88:8827
5sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityN©
 sequential/quant_conv2d_2/Conv2DConv2D7sequential/quant_conv2d_2/ste_sign_4/IdentityN:output:0>sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityN:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*
paddingSAME*
strides
2"
 sequential/quant_conv2d_2/Conv2D 
-sequential/batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-sequential/batch_normalization_2/LogicalAnd/x 
-sequential/batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-sequential/batch_normalization_2/LogicalAnd/yð
+sequential/batch_normalization_2/LogicalAnd
LogicalAnd6sequential/batch_normalization_2/LogicalAnd/x:output:06sequential/batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_2/LogicalAnd×
/sequential/batch_normalization_2/ReadVariableOpReadVariableOp8sequential_batch_normalization_2_readvariableop_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization_2/ReadVariableOpÝ
1sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:8*
dtype023
1sequential/batch_normalization_2/ReadVariableOp_1
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02B
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02D
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1³
1sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3)sequential/quant_conv2d_2/Conv2D:output:07sequential/batch_normalization_2/ReadVariableOp:value:09sequential/batch_normalization_2/ReadVariableOp_1:value:0Hsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ 8:8:8:8:8:*
epsilon%o:*
is_training( 23
1sequential/batch_normalization_2/FusedBatchNormV3
&sequential/batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2(
&sequential/batch_normalization_2/Const÷
"sequential/max_pooling2d_2/MaxPoolMaxPool5sequential/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPool
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
sequential/flatten/ConstÆ
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_2/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/flatten/Reshape°
&sequential/quant_dense/ste_sign_6/SignSign#sequential/flatten/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/quant_dense/ste_sign_6/Sign
'sequential/quant_dense/ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2)
'sequential/quant_dense/ste_sign_6/add/yè
%sequential/quant_dense/ste_sign_6/addAddV2*sequential/quant_dense/ste_sign_6/Sign:y:00sequential/quant_dense/ste_sign_6/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential/quant_dense/ste_sign_6/addº
(sequential/quant_dense/ste_sign_6/Sign_1Sign)sequential/quant_dense/ste_sign_6/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/quant_dense/ste_sign_6/Sign_1Å
*sequential/quant_dense/ste_sign_6/IdentityIdentity,sequential/quant_dense/ste_sign_6/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/quant_dense/ste_sign_6/Identity³
+sequential/quant_dense/ste_sign_6/IdentityN	IdentityN,sequential/quant_dense/ste_sign_6/Sign_1:y:0#sequential/flatten/Reshape:output:0*
T
2*,
_gradient_op_typeCustomGradient-181201*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2-
+sequential/quant_dense/ste_sign_6/IdentityNÔ
,sequential/quant_dense/MatMul/ReadVariableOpReadVariableOp5sequential_quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,sequential/quant_dense/MatMul/ReadVariableOpÇ
-sequential/quant_dense/MatMul/ste_sign_5/SignSign4sequential/quant_dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2/
-sequential/quant_dense/MatMul/ste_sign_5/Sign¥
.sequential/quant_dense/MatMul/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=20
.sequential/quant_dense/MatMul/ste_sign_5/add/yü
,sequential/quant_dense/MatMul/ste_sign_5/addAddV21sequential/quant_dense/MatMul/ste_sign_5/Sign:y:07sequential/quant_dense/MatMul/ste_sign_5/add/y:output:0*
T0* 
_output_shapes
:
2.
,sequential/quant_dense/MatMul/ste_sign_5/addÇ
/sequential/quant_dense/MatMul/ste_sign_5/Sign_1Sign0sequential/quant_dense/MatMul/ste_sign_5/add:z:0*
T0* 
_output_shapes
:
21
/sequential/quant_dense/MatMul/ste_sign_5/Sign_1Ò
1sequential/quant_dense/MatMul/ste_sign_5/IdentityIdentity3sequential/quant_dense/MatMul/ste_sign_5/Sign_1:y:0*
T0* 
_output_shapes
:
23
1sequential/quant_dense/MatMul/ste_sign_5/IdentityÉ
2sequential/quant_dense/MatMul/ste_sign_5/IdentityN	IdentityN3sequential/quant_dense/MatMul/ste_sign_5/Sign_1:y:04sequential/quant_dense/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-181211*,
_output_shapes
:
:
24
2sequential/quant_dense/MatMul/ste_sign_5/IdentityNî
sequential/quant_dense/MatMulMatMul4sequential/quant_dense/ste_sign_6/IdentityN:output:0;sequential/quant_dense/MatMul/ste_sign_5/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/quant_dense/MatMul 
-sequential/batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-sequential/batch_normalization_3/LogicalAnd/x 
-sequential/batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-sequential/batch_normalization_3/LogicalAnd/yð
+sequential/batch_normalization_3/LogicalAnd
LogicalAnd6sequential/batch_normalization_3/LogicalAnd/x:output:06sequential/batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_3/LogicalAndö
9sequential/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpBsequential_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02;
9sequential/batch_normalization_3/batchnorm/ReadVariableOp©
0sequential/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:22
0sequential/batch_normalization_3/batchnorm/add/y
.sequential/batch_normalization_3/batchnorm/addAddV2Asequential/batch_normalization_3/batchnorm/ReadVariableOp:value:09sequential/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:20
.sequential/batch_normalization_3/batchnorm/addÇ
0sequential/batch_normalization_3/batchnorm/RsqrtRsqrt2sequential/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:22
0sequential/batch_normalization_3/batchnorm/Rsqrt
=sequential/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpFsequential_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02?
=sequential/batch_normalization_3/batchnorm/mul/ReadVariableOp
.sequential/batch_normalization_3/batchnorm/mulMul4sequential/batch_normalization_3/batchnorm/Rsqrt:y:0Esequential/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:20
.sequential/batch_normalization_3/batchnorm/mulû
0sequential/batch_normalization_3/batchnorm/mul_1Mul'sequential/quant_dense/MatMul:product:02sequential/batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/batch_normalization_3/batchnorm/mul_1ü
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02=
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_1
0sequential/batch_normalization_3/batchnorm/mul_2MulCsequential/batch_normalization_3/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:22
0sequential/batch_normalization_3/batchnorm/mul_2ü
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpDsequential_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02=
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_2
.sequential/batch_normalization_3/batchnorm/subSubCsequential/batch_normalization_3/batchnorm/ReadVariableOp_2:value:04sequential/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:20
.sequential/batch_normalization_3/batchnorm/sub
0sequential/batch_normalization_3/batchnorm/add_1AddV24sequential/batch_normalization_3/batchnorm/mul_1:z:02sequential/batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/batch_normalization_3/batchnorm/add_1Å
(sequential/quant_dense_1/ste_sign_8/SignSign4sequential/batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/quant_dense_1/ste_sign_8/Sign
)sequential/quant_dense_1/ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)sequential/quant_dense_1/ste_sign_8/add/yð
'sequential/quant_dense_1/ste_sign_8/addAddV2,sequential/quant_dense_1/ste_sign_8/Sign:y:02sequential/quant_dense_1/ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/quant_dense_1/ste_sign_8/addÀ
*sequential/quant_dense_1/ste_sign_8/Sign_1Sign+sequential/quant_dense_1/ste_sign_8/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/quant_dense_1/ste_sign_8/Sign_1Ë
,sequential/quant_dense_1/ste_sign_8/IdentityIdentity.sequential/quant_dense_1/ste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential/quant_dense_1/ste_sign_8/IdentityÊ
-sequential/quant_dense_1/ste_sign_8/IdentityN	IdentityN.sequential/quant_dense_1/ste_sign_8/Sign_1:y:04sequential/batch_normalization_3/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-181239*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2/
-sequential/quant_dense_1/ste_sign_8/IdentityNÚ
.sequential/quant_dense_1/MatMul/ReadVariableOpReadVariableOp7sequential_quant_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype020
.sequential/quant_dense_1/MatMul/ReadVariableOpÍ
/sequential/quant_dense_1/MatMul/ste_sign_7/SignSign6sequential/quant_dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
21
/sequential/quant_dense_1/MatMul/ste_sign_7/Sign©
0sequential/quant_dense_1/MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=22
0sequential/quant_dense_1/MatMul/ste_sign_7/add/y
.sequential/quant_dense_1/MatMul/ste_sign_7/addAddV23sequential/quant_dense_1/MatMul/ste_sign_7/Sign:y:09sequential/quant_dense_1/MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
20
.sequential/quant_dense_1/MatMul/ste_sign_7/addÍ
1sequential/quant_dense_1/MatMul/ste_sign_7/Sign_1Sign2sequential/quant_dense_1/MatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
23
1sequential/quant_dense_1/MatMul/ste_sign_7/Sign_1Ø
3sequential/quant_dense_1/MatMul/ste_sign_7/IdentityIdentity5sequential/quant_dense_1/MatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
25
3sequential/quant_dense_1/MatMul/ste_sign_7/IdentityÑ
4sequential/quant_dense_1/MatMul/ste_sign_7/IdentityN	IdentityN5sequential/quant_dense_1/MatMul/ste_sign_7/Sign_1:y:06sequential/quant_dense_1/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-181249*,
_output_shapes
:
:
26
4sequential/quant_dense_1/MatMul/ste_sign_7/IdentityNö
sequential/quant_dense_1/MatMulMatMul6sequential/quant_dense_1/ste_sign_8/IdentityN:output:0=sequential/quant_dense_1/MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/quant_dense_1/MatMul 
-sequential/batch_normalization_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-sequential/batch_normalization_4/LogicalAnd/x 
-sequential/batch_normalization_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-sequential/batch_normalization_4/LogicalAnd/yð
+sequential/batch_normalization_4/LogicalAnd
LogicalAnd6sequential/batch_normalization_4/LogicalAnd/x:output:06sequential/batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_4/LogicalAndö
9sequential/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpBsequential_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02;
9sequential/batch_normalization_4/batchnorm/ReadVariableOp©
0sequential/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:22
0sequential/batch_normalization_4/batchnorm/add/y
.sequential/batch_normalization_4/batchnorm/addAddV2Asequential/batch_normalization_4/batchnorm/ReadVariableOp:value:09sequential/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:20
.sequential/batch_normalization_4/batchnorm/addÇ
0sequential/batch_normalization_4/batchnorm/RsqrtRsqrt2sequential/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:22
0sequential/batch_normalization_4/batchnorm/Rsqrt
=sequential/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpFsequential_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02?
=sequential/batch_normalization_4/batchnorm/mul/ReadVariableOp
.sequential/batch_normalization_4/batchnorm/mulMul4sequential/batch_normalization_4/batchnorm/Rsqrt:y:0Esequential/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:20
.sequential/batch_normalization_4/batchnorm/mulý
0sequential/batch_normalization_4/batchnorm/mul_1Mul)sequential/quant_dense_1/MatMul:product:02sequential/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/batch_normalization_4/batchnorm/mul_1ü
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02=
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1
0sequential/batch_normalization_4/batchnorm/mul_2MulCsequential/batch_normalization_4/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:22
0sequential/batch_normalization_4/batchnorm/mul_2ü
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpDsequential_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02=
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_2
.sequential/batch_normalization_4/batchnorm/subSubCsequential/batch_normalization_4/batchnorm/ReadVariableOp_2:value:04sequential/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:20
.sequential/batch_normalization_4/batchnorm/sub
0sequential/batch_normalization_4/batchnorm/add_1AddV24sequential/batch_normalization_4/batchnorm/mul_1:z:02sequential/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/batch_normalization_4/batchnorm/add_1Ç
)sequential/quant_dense_2/ste_sign_10/SignSign4sequential/batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/quant_dense_2/ste_sign_10/Sign
*sequential/quant_dense_2/ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2,
*sequential/quant_dense_2/ste_sign_10/add/yô
(sequential/quant_dense_2/ste_sign_10/addAddV2-sequential/quant_dense_2/ste_sign_10/Sign:y:03sequential/quant_dense_2/ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/quant_dense_2/ste_sign_10/addÃ
+sequential/quant_dense_2/ste_sign_10/Sign_1Sign,sequential/quant_dense_2/ste_sign_10/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential/quant_dense_2/ste_sign_10/Sign_1Î
-sequential/quant_dense_2/ste_sign_10/IdentityIdentity/sequential/quant_dense_2/ste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/quant_dense_2/ste_sign_10/IdentityÍ
.sequential/quant_dense_2/ste_sign_10/IdentityN	IdentityN/sequential/quant_dense_2/ste_sign_10/Sign_1:y:04sequential/batch_normalization_4/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-181277*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ20
.sequential/quant_dense_2/ste_sign_10/IdentityNÙ
.sequential/quant_dense_2/MatMul/ReadVariableOpReadVariableOp7sequential_quant_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype020
.sequential/quant_dense_2/MatMul/ReadVariableOpÌ
/sequential/quant_dense_2/MatMul/ste_sign_9/SignSign6sequential/quant_dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	21
/sequential/quant_dense_2/MatMul/ste_sign_9/Sign©
0sequential/quant_dense_2/MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=22
0sequential/quant_dense_2/MatMul/ste_sign_9/add/y
.sequential/quant_dense_2/MatMul/ste_sign_9/addAddV23sequential/quant_dense_2/MatMul/ste_sign_9/Sign:y:09sequential/quant_dense_2/MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	20
.sequential/quant_dense_2/MatMul/ste_sign_9/addÌ
1sequential/quant_dense_2/MatMul/ste_sign_9/Sign_1Sign2sequential/quant_dense_2/MatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	23
1sequential/quant_dense_2/MatMul/ste_sign_9/Sign_1×
3sequential/quant_dense_2/MatMul/ste_sign_9/IdentityIdentity5sequential/quant_dense_2/MatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	25
3sequential/quant_dense_2/MatMul/ste_sign_9/IdentityÏ
4sequential/quant_dense_2/MatMul/ste_sign_9/IdentityN	IdentityN5sequential/quant_dense_2/MatMul/ste_sign_9/Sign_1:y:06sequential/quant_dense_2/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-181287**
_output_shapes
:	:	26
4sequential/quant_dense_2/MatMul/ste_sign_9/IdentityNö
sequential/quant_dense_2/MatMulMatMul7sequential/quant_dense_2/ste_sign_10/IdentityN:output:0=sequential/quant_dense_2/MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/quant_dense_2/MatMul 
-sequential/batch_normalization_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-sequential/batch_normalization_5/LogicalAnd/x 
-sequential/batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-sequential/batch_normalization_5/LogicalAnd/yð
+sequential/batch_normalization_5/LogicalAnd
LogicalAnd6sequential/batch_normalization_5/LogicalAnd/x:output:06sequential/batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_5/LogicalAndõ
9sequential/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOpBsequential_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential/batch_normalization_5/batchnorm/ReadVariableOp©
0sequential/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:22
0sequential/batch_normalization_5/batchnorm/add/y
.sequential/batch_normalization_5/batchnorm/addAddV2Asequential/batch_normalization_5/batchnorm/ReadVariableOp:value:09sequential/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:20
.sequential/batch_normalization_5/batchnorm/addÆ
0sequential/batch_normalization_5/batchnorm/RsqrtRsqrt2sequential/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:22
0sequential/batch_normalization_5/batchnorm/Rsqrt
=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpFsequential_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOp
.sequential/batch_normalization_5/batchnorm/mulMul4sequential/batch_normalization_5/batchnorm/Rsqrt:y:0Esequential/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.sequential/batch_normalization_5/batchnorm/mulü
0sequential/batch_normalization_5/batchnorm/mul_1Mul)sequential/quant_dense_2/MatMul:product:02sequential/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/batch_normalization_5/batchnorm/mul_1û
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1
0sequential/batch_normalization_5/batchnorm/mul_2MulCsequential/batch_normalization_5/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:22
0sequential/batch_normalization_5/batchnorm/mul_2û
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpDsequential_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02=
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2
.sequential/batch_normalization_5/batchnorm/subSubCsequential/batch_normalization_5/batchnorm/ReadVariableOp_2:value:04sequential/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization_5/batchnorm/sub
0sequential/batch_normalization_5/batchnorm/add_1AddV24sequential/batch_normalization_5/batchnorm/mul_1:z:02sequential/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/batch_normalization_5/batchnorm/add_1±
sequential/activation/SoftmaxSoftmax4sequential/batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/activation/SoftmaxÊ
IdentityIdentity'sequential/activation/Softmax:softmax:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1A^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_2/ReadVariableOp2^sequential/batch_normalization_2/ReadVariableOp_1:^sequential/batch_normalization_3/batchnorm/ReadVariableOp<^sequential/batch_normalization_3/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_3/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_3/batchnorm/mul/ReadVariableOp:^sequential/batch_normalization_4/batchnorm/ReadVariableOp<^sequential/batch_normalization_4/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_4/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_4/batchnorm/mul/ReadVariableOp:^sequential/batch_normalization_5/batchnorm/ReadVariableOp<^sequential/batch_normalization_5/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_5/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_5/batchnorm/mul/ReadVariableOp.^sequential/quant_conv2d/Conv2D/ReadVariableOp0^sequential/quant_conv2d_1/Conv2D/ReadVariableOp0^sequential/quant_conv2d_2/Conv2D/ReadVariableOp-^sequential/quant_dense/MatMul/ReadVariableOp/^sequential/quant_dense_1/MatMul/ReadVariableOp/^sequential/quant_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::2
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_2/ReadVariableOp/sequential/batch_normalization_2/ReadVariableOp2f
1sequential/batch_normalization_2/ReadVariableOp_11sequential/batch_normalization_2/ReadVariableOp_12v
9sequential/batch_normalization_3/batchnorm/ReadVariableOp9sequential/batch_normalization_3/batchnorm/ReadVariableOp2z
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_1;sequential/batch_normalization_3/batchnorm/ReadVariableOp_12z
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_2;sequential/batch_normalization_3/batchnorm/ReadVariableOp_22~
=sequential/batch_normalization_3/batchnorm/mul/ReadVariableOp=sequential/batch_normalization_3/batchnorm/mul/ReadVariableOp2v
9sequential/batch_normalization_4/batchnorm/ReadVariableOp9sequential/batch_normalization_4/batchnorm/ReadVariableOp2z
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1;sequential/batch_normalization_4/batchnorm/ReadVariableOp_12z
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_2;sequential/batch_normalization_4/batchnorm/ReadVariableOp_22~
=sequential/batch_normalization_4/batchnorm/mul/ReadVariableOp=sequential/batch_normalization_4/batchnorm/mul/ReadVariableOp2v
9sequential/batch_normalization_5/batchnorm/ReadVariableOp9sequential/batch_normalization_5/batchnorm/ReadVariableOp2z
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1;sequential/batch_normalization_5/batchnorm/ReadVariableOp_12z
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2;sequential/batch_normalization_5/batchnorm/ReadVariableOp_22~
=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOp=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOp2^
-sequential/quant_conv2d/Conv2D/ReadVariableOp-sequential/quant_conv2d/Conv2D/ReadVariableOp2b
/sequential/quant_conv2d_1/Conv2D/ReadVariableOp/sequential/quant_conv2d_1/Conv2D/ReadVariableOp2b
/sequential/quant_conv2d_2/Conv2D/ReadVariableOp/sequential/quant_conv2d_2/Conv2D/ReadVariableOp2\
,sequential/quant_dense/MatMul/ReadVariableOp,sequential/quant_dense/MatMul/ReadVariableOp2`
.sequential/quant_dense_1/MatMul/ReadVariableOp.sequential/quant_dense_1/MatMul/ReadVariableOp2`
.sequential/quant_dense_2/MatMul/ReadVariableOp.sequential/quant_dense_2/MatMul/ReadVariableOp:d `
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namequant_conv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ï
b
D__inference_ste_sign_layer_call_and_return_conditional_losses_185193

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:82
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:82
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:82
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:82

Identity¬
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-185184*8
_output_shapes&
$:8:82
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:82

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:8:N J
&
_output_shapes
:8
 
_user_specified_nameinputs

¥
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_181556

inputs"
conv2d_readvariableop_resource
identity¢Conv2D/ReadVariableOp³
ste_sign_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_1815222
ste_sign_2/PartitionedCall§
ste_sign_2/IdentityIdentity#ste_sign_2/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
ste_sign_2/Identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02
Conv2D/ReadVariableOp½
!Conv2D/ste_sign_1/PartitionedCallPartitionedCallConv2D/ReadVariableOp:value:0*
Tin
2*
Tout
2*&
_output_shapes
:88*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_1815452#
!Conv2D/ste_sign_1/PartitionedCall¡
Conv2D/ste_sign_1/IdentityIdentity*Conv2D/ste_sign_1/PartitionedCall:output:0*
T0*&
_output_shapes
:882
Conv2D/ste_sign_1/IdentityÑ
Conv2DConv2Dste_sign_2/Identity:output:0#Conv2D/ste_sign_1/Identity:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*
paddingSAME*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: 
Ú
¯$
__inference__traced_save_185515
file_prefix2
.savev2_quant_conv2d_kernel_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop4
0savev2_quant_conv2d_1_kernel_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop4
0savev2_quant_conv2d_2_kernel_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop1
-savev2_quant_dense_kernel_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop3
/savev2_quant_dense_1_kernel_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop3
/savev2_quant_dense_2_kernel_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_quant_conv2d_kernel_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop;
7savev2_adam_quant_conv2d_1_kernel_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop;
7savev2_adam_quant_conv2d_2_kernel_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop8
4savev2_adam_quant_dense_kernel_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop:
6savev2_adam_quant_dense_1_kernel_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop:
6savev2_adam_quant_dense_2_kernel_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop9
5savev2_adam_quant_conv2d_kernel_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop;
7savev2_adam_quant_conv2d_1_kernel_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop;
7savev2_adam_quant_conv2d_2_kernel_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop8
4savev2_adam_quant_dense_kernel_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop:
6savev2_adam_quant_dense_1_kernel_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop:
6savev2_adam_quant_dense_2_kernel_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1¥
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b727caa36de9416db18bf4246fb64193/part2
StringJoin/inputs_1

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÉ)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Û(
valueÑ(BÎ(KB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names¡
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices÷"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_quant_conv2d_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop0savev2_quant_conv2d_1_kernel_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop0savev2_quant_conv2d_2_kernel_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop-savev2_quant_dense_kernel_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop/savev2_quant_dense_1_kernel_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop/savev2_quant_dense_2_kernel_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_quant_conv2d_kernel_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop7savev2_adam_quant_conv2d_1_kernel_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop7savev2_adam_quant_conv2d_2_kernel_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop4savev2_adam_quant_dense_kernel_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop6savev2_adam_quant_dense_1_kernel_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop6savev2_adam_quant_dense_2_kernel_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop5savev2_adam_quant_conv2d_kernel_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop7savev2_adam_quant_conv2d_1_kernel_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop7savev2_adam_quant_conv2d_2_kernel_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop4savev2_adam_quant_dense_kernel_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop6savev2_adam_quant_dense_1_kernel_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop6savev2_adam_quant_dense_2_kernel_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*æ
_input_shapesÔ
Ñ: :8:8:8:8:8:88:8:8:8:8:88:8:8:8:8:
:::::
:::::	::::: : : : : : : : : :8:8:8:88:8:8:88:8:8:
:::
:::	:::8:8:8:88:8:8:88:8:8:
:::
:::	::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:8: 

_output_shapes
:8: 

_output_shapes
:8: 

_output_shapes
:8: 

_output_shapes
:8:,(
&
_output_shapes
:88: 

_output_shapes
:8: 

_output_shapes
:8: 	

_output_shapes
:8: 


_output_shapes
:8:,(
&
_output_shapes
:88: 

_output_shapes
:8: 

_output_shapes
:8: 

_output_shapes
:8: 

_output_shapes
:8:&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :,((
&
_output_shapes
:8: )

_output_shapes
:8: *

_output_shapes
:8:,+(
&
_output_shapes
:88: ,

_output_shapes
:8: -

_output_shapes
:8:,.(
&
_output_shapes
:88: /

_output_shapes
:8: 0

_output_shapes
:8:&1"
 
_output_shapes
:
:!2

_output_shapes	
::!3

_output_shapes	
::&4"
 
_output_shapes
:
:!5

_output_shapes	
::!6

_output_shapes	
::%7!

_output_shapes
:	: 8

_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:8: ;

_output_shapes
:8: <

_output_shapes
:8:,=(
&
_output_shapes
:88: >

_output_shapes
:8: ?

_output_shapes
:8:,@(
&
_output_shapes
:88: A

_output_shapes
:8: B

_output_shapes
:8:&C"
 
_output_shapes
:
:!D

_output_shapes	
::!E

_output_shapes	
::&F"
 
_output_shapes
:
:!G

_output_shapes	
::!H

_output_shapes	
::%I!

_output_shapes
:	: J

_output_shapes
:: K

_output_shapes
::L

_output_shapes
: 

u
/__inference_quant_conv2d_2_layer_call_fn_181774

inputs
unknown
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1817662
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: 
õ%

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184568

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_184553
assignmovingavg_1_184560
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/184553*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/184553*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_184553*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/184553*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/184553*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_184553AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/184553*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/184560*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184560*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_184560*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184560*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184560*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_184560AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/184560*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¸
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ì
§
4__inference_batch_normalization_layer_call_fn_184251

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1814482
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó%

O__inference_batch_normalization_layer_call_and_return_conditional_losses_184216

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_184201
assignmovingavg_1_184208
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/184201*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/184201*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_184201*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/184201*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/184201*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_184201AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/184201*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/184208*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184208*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_184208*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184208*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184208*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_184208AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/184208*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¸
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¼
b
F__inference_activation_layer_call_and_return_conditional_losses_185176

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÒY
µ
F__inference_sequential_layer_call_and_return_conditional_losses_183220

inputs
quant_conv2d_183142
batch_normalization_183145
batch_normalization_183147
batch_normalization_183149
batch_normalization_183151
quant_conv2d_1_183155 
batch_normalization_1_183158 
batch_normalization_1_183160 
batch_normalization_1_183162 
batch_normalization_1_183164
quant_conv2d_2_183168 
batch_normalization_2_183171 
batch_normalization_2_183173 
batch_normalization_2_183175 
batch_normalization_2_183177
quant_dense_183182 
batch_normalization_3_183185 
batch_normalization_3_183187 
batch_normalization_3_183189 
batch_normalization_3_183191
quant_dense_1_183194 
batch_normalization_4_183197 
batch_normalization_4_183199 
batch_normalization_4_183201 
batch_normalization_4_183203
quant_dense_2_183206 
batch_normalization_5_183209 
batch_normalization_5_183211 
batch_normalization_5_183213 
batch_normalization_5_183215
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢$quant_conv2d/StatefulPartitionedCall¢&quant_conv2d_1/StatefulPartitionedCall¢&quant_conv2d_2/StatefulPartitionedCall¢#quant_dense/StatefulPartitionedCall¢%quant_dense_1/StatefulPartitionedCall¢%quant_dense_2/StatefulPartitionedCallÖ
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsquant_conv2d_183142*
Tin
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1813462&
$quant_conv2d/StatefulPartitionedCallð
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_183145batch_normalization_183147batch_normalization_183149batch_normalization_183151*
Tin	
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1824442-
+batch_normalization/StatefulPartitionedCallØ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1815002
max_pooling2d/PartitionedCallý
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_183155*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1815562(
&quant_conv2d_1/StatefulPartitionedCallÿ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_183158batch_normalization_1_183160batch_normalization_1_183162batch_normalization_1_183164*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825392/
-batch_normalization_1/StatefulPartitionedCallà
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1817102!
max_pooling2d_1/PartitionedCallÿ
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_183168*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1817662(
&quant_conv2d_2/StatefulPartitionedCallÿ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_183171batch_normalization_2_183173batch_normalization_2_183175batch_normalization_2_183177*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1826342/
-batch_normalization_2/StatefulPartitionedCallà
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1819202!
max_pooling2d_2/PartitionedCall³
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1826772
flatten/PartitionedCallä
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_183182*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_1827082%
#quant_dense/StatefulPartitionedCallõ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_183185batch_normalization_3_183187batch_normalization_3_183189batch_normalization_3_183191*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1820672/
-batch_normalization_3/StatefulPartitionedCall
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_183194*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1827782'
%quant_dense_1/StatefulPartitionedCall÷
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_183197batch_normalization_4_183199batch_normalization_4_183201batch_normalization_4_183203*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1822192/
-batch_normalization_4/StatefulPartitionedCall
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_183206*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1828482'
%quant_dense_2/StatefulPartitionedCallö
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_183209batch_normalization_5_183211batch_normalization_5_183213batch_normalization_5_183215*
Tin	
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1823712/
-batch_normalization_5/StatefulPartitionedCallÉ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1829002
activation/PartitionedCall
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

u
/__inference_quant_conv2d_1_layer_call_fn_181564

inputs
unknown
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1815562
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: 

©
6__inference_batch_normalization_1_layer_call_fn_184509

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿA
8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¤
t
.__inference_quant_dense_1_layer_call_fn_184893

inputs
unknown
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1827782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ÀÈ
Ý,
"__inference__traced_restore_185752
file_prefix(
$assignvariableop_quant_conv2d_kernel0
,assignvariableop_1_batch_normalization_gamma/
+assignvariableop_2_batch_normalization_beta6
2assignvariableop_3_batch_normalization_moving_mean:
6assignvariableop_4_batch_normalization_moving_variance,
(assignvariableop_5_quant_conv2d_1_kernel2
.assignvariableop_6_batch_normalization_1_gamma1
-assignvariableop_7_batch_normalization_1_beta8
4assignvariableop_8_batch_normalization_1_moving_mean<
8assignvariableop_9_batch_normalization_1_moving_variance-
)assignvariableop_10_quant_conv2d_2_kernel3
/assignvariableop_11_batch_normalization_2_gamma2
.assignvariableop_12_batch_normalization_2_beta9
5assignvariableop_13_batch_normalization_2_moving_mean=
9assignvariableop_14_batch_normalization_2_moving_variance*
&assignvariableop_15_quant_dense_kernel3
/assignvariableop_16_batch_normalization_3_gamma2
.assignvariableop_17_batch_normalization_3_beta9
5assignvariableop_18_batch_normalization_3_moving_mean=
9assignvariableop_19_batch_normalization_3_moving_variance,
(assignvariableop_20_quant_dense_1_kernel3
/assignvariableop_21_batch_normalization_4_gamma2
.assignvariableop_22_batch_normalization_4_beta9
5assignvariableop_23_batch_normalization_4_moving_mean=
9assignvariableop_24_batch_normalization_4_moving_variance,
(assignvariableop_25_quant_dense_2_kernel3
/assignvariableop_26_batch_normalization_5_gamma2
.assignvariableop_27_batch_normalization_5_beta9
5assignvariableop_28_batch_normalization_5_moving_mean=
9assignvariableop_29_batch_normalization_5_moving_variance!
assignvariableop_30_adam_iter#
assignvariableop_31_adam_beta_1#
assignvariableop_32_adam_beta_2"
assignvariableop_33_adam_decay*
&assignvariableop_34_adam_learning_rate
assignvariableop_35_total
assignvariableop_36_count
assignvariableop_37_total_1
assignvariableop_38_count_12
.assignvariableop_39_adam_quant_conv2d_kernel_m8
4assignvariableop_40_adam_batch_normalization_gamma_m7
3assignvariableop_41_adam_batch_normalization_beta_m4
0assignvariableop_42_adam_quant_conv2d_1_kernel_m:
6assignvariableop_43_adam_batch_normalization_1_gamma_m9
5assignvariableop_44_adam_batch_normalization_1_beta_m4
0assignvariableop_45_adam_quant_conv2d_2_kernel_m:
6assignvariableop_46_adam_batch_normalization_2_gamma_m9
5assignvariableop_47_adam_batch_normalization_2_beta_m1
-assignvariableop_48_adam_quant_dense_kernel_m:
6assignvariableop_49_adam_batch_normalization_3_gamma_m9
5assignvariableop_50_adam_batch_normalization_3_beta_m3
/assignvariableop_51_adam_quant_dense_1_kernel_m:
6assignvariableop_52_adam_batch_normalization_4_gamma_m9
5assignvariableop_53_adam_batch_normalization_4_beta_m3
/assignvariableop_54_adam_quant_dense_2_kernel_m:
6assignvariableop_55_adam_batch_normalization_5_gamma_m9
5assignvariableop_56_adam_batch_normalization_5_beta_m2
.assignvariableop_57_adam_quant_conv2d_kernel_v8
4assignvariableop_58_adam_batch_normalization_gamma_v7
3assignvariableop_59_adam_batch_normalization_beta_v4
0assignvariableop_60_adam_quant_conv2d_1_kernel_v:
6assignvariableop_61_adam_batch_normalization_1_gamma_v9
5assignvariableop_62_adam_batch_normalization_1_beta_v4
0assignvariableop_63_adam_quant_conv2d_2_kernel_v:
6assignvariableop_64_adam_batch_normalization_2_gamma_v9
5assignvariableop_65_adam_batch_normalization_2_beta_v1
-assignvariableop_66_adam_quant_dense_kernel_v:
6assignvariableop_67_adam_batch_normalization_3_gamma_v9
5assignvariableop_68_adam_batch_normalization_3_beta_v3
/assignvariableop_69_adam_quant_dense_1_kernel_v:
6assignvariableop_70_adam_batch_normalization_4_gamma_v9
5assignvariableop_71_adam_batch_normalization_4_beta_v3
/assignvariableop_72_adam_quant_dense_2_kernel_v:
6assignvariableop_73_adam_batch_normalization_5_gamma_v9
5assignvariableop_74_adam_batch_normalization_5_beta_v
identity_76¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1Ï)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Û(
valueÑ(BÎ(KB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names§
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¥
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp$assignvariableop_quant_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOp,assignvariableop_1_batch_normalization_gammaIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2¡
AssignVariableOp_2AssignVariableOp+assignvariableop_2_batch_normalization_betaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3¨
AssignVariableOp_3AssignVariableOp2assignvariableop_3_batch_normalization_moving_meanIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4¬
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_moving_varianceIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp(assignvariableop_5_quant_conv2d_1_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8ª
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9®
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10¢
AssignVariableOp_10AssignVariableOp)assignvariableop_10_quant_conv2d_2_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11¨
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_2_gammaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_2_betaIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13®
AssignVariableOp_13AssignVariableOp5assignvariableop_13_batch_normalization_2_moving_meanIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14²
AssignVariableOp_14AssignVariableOp9assignvariableop_14_batch_normalization_2_moving_varianceIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOp&assignvariableop_15_quant_dense_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16¨
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_3_gammaIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17§
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_3_betaIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_3_moving_meanIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19²
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_3_moving_varianceIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20¡
AssignVariableOp_20AssignVariableOp(assignvariableop_20_quant_dense_1_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21¨
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_4_gammaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22§
AssignVariableOp_22AssignVariableOp.assignvariableop_22_batch_normalization_4_betaIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23®
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_4_moving_meanIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_4_moving_varianceIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25¡
AssignVariableOp_25AssignVariableOp(assignvariableop_25_quant_dense_2_kernelIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26¨
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_5_gammaIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27§
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_5_betaIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28®
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_5_moving_meanIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29²
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_5_moving_varianceIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0	*
_output_shapes
:2
Identity_30
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_iterIdentity_30:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_beta_1Identity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_beta_2Identity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_decayIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_learning_rateIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_1Identity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_1Identity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39§
AssignVariableOp_39AssignVariableOp.assignvariableop_39_adam_quant_conv2d_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40­
AssignVariableOp_40AssignVariableOp4assignvariableop_40_adam_batch_normalization_gamma_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41¬
AssignVariableOp_41AssignVariableOp3assignvariableop_41_adam_batch_normalization_beta_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42©
AssignVariableOp_42AssignVariableOp0assignvariableop_42_adam_quant_conv2d_1_kernel_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43¯
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_1_gamma_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44®
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_batch_normalization_1_beta_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45©
AssignVariableOp_45AssignVariableOp0assignvariableop_45_adam_quant_conv2d_2_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46¯
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_2_gamma_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47®
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adam_batch_normalization_2_beta_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48¦
AssignVariableOp_48AssignVariableOp-assignvariableop_48_adam_quant_dense_kernel_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49¯
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_3_gamma_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50®
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_3_beta_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51¨
AssignVariableOp_51AssignVariableOp/assignvariableop_51_adam_quant_dense_1_kernel_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52¯
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_batch_normalization_4_gamma_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53®
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adam_batch_normalization_4_beta_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54¨
AssignVariableOp_54AssignVariableOp/assignvariableop_54_adam_quant_dense_2_kernel_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55¯
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_5_gamma_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56®
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_batch_normalization_5_beta_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57§
AssignVariableOp_57AssignVariableOp.assignvariableop_57_adam_quant_conv2d_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58­
AssignVariableOp_58AssignVariableOp4assignvariableop_58_adam_batch_normalization_gamma_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59¬
AssignVariableOp_59AssignVariableOp3assignvariableop_59_adam_batch_normalization_beta_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60©
AssignVariableOp_60AssignVariableOp0assignvariableop_60_adam_quant_conv2d_1_kernel_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61¯
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_1_gamma_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62®
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_1_beta_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63©
AssignVariableOp_63AssignVariableOp0assignvariableop_63_adam_quant_conv2d_2_kernel_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64¯
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_2_gamma_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65®
AssignVariableOp_65AssignVariableOp5assignvariableop_65_adam_batch_normalization_2_beta_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66¦
AssignVariableOp_66AssignVariableOp-assignvariableop_66_adam_quant_dense_kernel_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67¯
AssignVariableOp_67AssignVariableOp6assignvariableop_67_adam_batch_normalization_3_gamma_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68®
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_batch_normalization_3_beta_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69¨
AssignVariableOp_69AssignVariableOp/assignvariableop_69_adam_quant_dense_1_kernel_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70¯
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_4_gamma_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71®
AssignVariableOp_71AssignVariableOp5assignvariableop_71_adam_batch_normalization_4_beta_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72¨
AssignVariableOp_72AssignVariableOp/assignvariableop_72_adam_quant_dense_2_kernel_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73¯
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_batch_normalization_5_gamma_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74®
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_batch_normalization_5_beta_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÐ
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75Ý
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*Ã
_input_shapes±
®: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: :J

_output_shapes
: :K

_output_shapes
: 
ô
ô
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184672

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ 8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstÚ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ 8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¾
G
+__inference_ste_sign_4_layer_call_fn_185266

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_1817322
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs

§
4__inference_batch_normalization_layer_call_fn_184333

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1824222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ8::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ñ
d
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_181545

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:882
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:882
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:882
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:882

Identity¬
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-181536*8
_output_shapes&
$:88:882
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:882

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:88:N J
&
_output_shapes
:88
 
_user_specified_nameinputs
ú

Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_185145

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Û
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

©
6__inference_batch_normalization_1_layer_call_fn_184522

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1825392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿA
8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¯%

O__inference_batch_normalization_layer_call_and_return_conditional_losses_184298

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_184283
assignmovingavg_1_184290
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/184283*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/184283*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_184283*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/184283*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/184283*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_184283AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/184283*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/184290*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184290*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_184290*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184290*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184290*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_184290AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/184290*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp§
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ì
©
6__inference_batch_normalization_4_layer_call_fn_185004

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1821832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
è
©
6__inference_batch_normalization_5_layer_call_fn_185171

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1823712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

¤
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_184886

inputs"
matmul_readvariableop_resource
identity¢MatMul/ReadVariableOpe
ste_sign_8/SignSigninputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_8/Signi
ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
ste_sign_8/add/y
ste_sign_8/addAddV2ste_sign_8/Sign:y:0ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_8/addu
ste_sign_8/Sign_1Signste_sign_8/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_8/Sign_1
ste_sign_8/IdentityIdentityste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ste_sign_8/IdentityÑ
ste_sign_8/IdentityN	IdentityNste_sign_8/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-184866*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2
ste_sign_8/IdentityN
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOp
MatMul/ste_sign_7/SignSignMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_7/Signw
MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
MatMul/ste_sign_7/add/y 
MatMul/ste_sign_7/addAddV2MatMul/ste_sign_7/Sign:y:0 MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_7/add
MatMul/ste_sign_7/Sign_1SignMatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_7/Sign_1
MatMul/ste_sign_7/IdentityIdentityMatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
2
MatMul/ste_sign_7/Identityí
MatMul/ste_sign_7/IdentityN	IdentityNMatMul/ste_sign_7/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-184876*,
_output_shapes
:
:
2
MatMul/ste_sign_7/IdentityN
MatMulMatMulste_sign_8/IdentityN:output:0$MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
õ%

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_181658

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_181643
assignmovingavg_1_181650
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/181643*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/181643*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_181643*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/181643*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/181643*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_181643AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/181643*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/181650*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/181650*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_181650*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/181650*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/181650*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_181650AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/181650*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¸
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¼
ô
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_181693

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¼
b
F__inference_activation_layer_call_and_return_conditional_losses_182900

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö¹
×
F__inference_sequential_layer_call_and_return_conditional_losses_184040

inputs/
+quant_conv2d_conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource1
-quant_conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource1
-quant_conv2d_2_conv2d_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource.
*quant_dense_matmul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource=
9batch_normalization_3_batchnorm_readvariableop_1_resource=
9batch_normalization_3_batchnorm_readvariableop_2_resource0
,quant_dense_1_matmul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource0
,quant_dense_2_matmul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource?
;batch_normalization_5_batchnorm_mul_readvariableop_resource=
9batch_normalization_5_batchnorm_readvariableop_1_resource=
9batch_normalization_5_batchnorm_readvariableop_2_resource
identity¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢.batch_normalization_3/batchnorm/ReadVariableOp¢0batch_normalization_3/batchnorm/ReadVariableOp_1¢0batch_normalization_3/batchnorm/ReadVariableOp_2¢2batch_normalization_3/batchnorm/mul/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢0batch_normalization_4/batchnorm/ReadVariableOp_1¢0batch_normalization_4/batchnorm/ReadVariableOp_2¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢0batch_normalization_5/batchnorm/ReadVariableOp_1¢0batch_normalization_5/batchnorm/ReadVariableOp_2¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢"quant_conv2d/Conv2D/ReadVariableOp¢$quant_conv2d_1/Conv2D/ReadVariableOp¢$quant_conv2d_2/Conv2D/ReadVariableOp¢!quant_dense/MatMul/ReadVariableOp¢#quant_dense_1/MatMul/ReadVariableOp¢#quant_dense_2/MatMul/ReadVariableOp¼
"quant_conv2d/Conv2D/ReadVariableOpReadVariableOp+quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02$
"quant_conv2d/Conv2D/ReadVariableOp«
!quant_conv2d/Conv2D/ste_sign/SignSign*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:82#
!quant_conv2d/Conv2D/ste_sign/Sign
"quant_conv2d/Conv2D/ste_sign/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2$
"quant_conv2d/Conv2D/ste_sign/add/yÒ
 quant_conv2d/Conv2D/ste_sign/addAddV2%quant_conv2d/Conv2D/ste_sign/Sign:y:0+quant_conv2d/Conv2D/ste_sign/add/y:output:0*
T0*&
_output_shapes
:82"
 quant_conv2d/Conv2D/ste_sign/add©
#quant_conv2d/Conv2D/ste_sign/Sign_1Sign$quant_conv2d/Conv2D/ste_sign/add:z:0*
T0*&
_output_shapes
:82%
#quant_conv2d/Conv2D/ste_sign/Sign_1´
%quant_conv2d/Conv2D/ste_sign/IdentityIdentity'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*
T0*&
_output_shapes
:82'
%quant_conv2d/Conv2D/ste_sign/Identity§
&quant_conv2d/Conv2D/ste_sign/IdentityN	IdentityN'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183818*8
_output_shapes&
$:8:82(
&quant_conv2d/Conv2D/ste_sign/IdentityNÐ
quant_conv2d/Conv2DConv2Dinputs/quant_conv2d/Conv2D/ste_sign/IdentityN:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
paddingSAME*
strides
2
quant_conv2d/Conv2D
 batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 batch_normalization/LogicalAnd/x
 batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 batch_normalization/LogicalAnd/y¼
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2 
batch_normalization/LogicalAnd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ù
$batch_normalization/FusedBatchNormV3FusedBatchNormV3quant_conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3{
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization/ConstÐ
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool¢
quant_conv2d_1/ste_sign_2/SignSignmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82 
quant_conv2d_1/ste_sign_2/Sign
quant_conv2d_1/ste_sign_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2!
quant_conv2d_1/ste_sign_2/add/yÏ
quant_conv2d_1/ste_sign_2/addAddV2"quant_conv2d_1/ste_sign_2/Sign:y:0(quant_conv2d_1/ste_sign_2/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82
quant_conv2d_1/ste_sign_2/add©
 quant_conv2d_1/ste_sign_2/Sign_1Sign!quant_conv2d_1/ste_sign_2/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82"
 quant_conv2d_1/ste_sign_2/Sign_1´
"quant_conv2d_1/ste_sign_2/IdentityIdentity$quant_conv2d_1/ste_sign_2/Sign_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
82$
"quant_conv2d_1/ste_sign_2/Identity¤
#quant_conv2d_1/ste_sign_2/IdentityN	IdentityN$quant_conv2d_1/ste_sign_2/Sign_1:y:0max_pooling2d/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-183846*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿA
8:ÿÿÿÿÿÿÿÿÿA
82%
#quant_conv2d_1/ste_sign_2/IdentityNÂ
$quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_1/Conv2D/ReadVariableOpµ
%quant_conv2d_1/Conv2D/ste_sign_1/SignSign,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:882'
%quant_conv2d_1/Conv2D/ste_sign_1/Sign
&quant_conv2d_1/Conv2D/ste_sign_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2(
&quant_conv2d_1/Conv2D/ste_sign_1/add/yâ
$quant_conv2d_1/Conv2D/ste_sign_1/addAddV2)quant_conv2d_1/Conv2D/ste_sign_1/Sign:y:0/quant_conv2d_1/Conv2D/ste_sign_1/add/y:output:0*
T0*&
_output_shapes
:882&
$quant_conv2d_1/Conv2D/ste_sign_1/addµ
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1Sign(quant_conv2d_1/Conv2D/ste_sign_1/add:z:0*
T0*&
_output_shapes
:882)
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1À
)quant_conv2d_1/Conv2D/ste_sign_1/IdentityIdentity+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0*
T0*&
_output_shapes
:882+
)quant_conv2d_1/Conv2D/ste_sign_1/Identityµ
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityN	IdentityN+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183856*8
_output_shapes&
$:88:882,
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityNý
quant_conv2d_1/Conv2DConv2D,quant_conv2d_1/ste_sign_2/IdentityN:output:03quant_conv2d_1/Conv2D/ste_sign_1/IdentityN:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
8*
paddingSAME*
strides
2
quant_conv2d_1/Conv2D
"batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_1/LogicalAnd/x
"batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_1/LogicalAnd/yÄ
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_1/LogicalAnd¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:8*
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:8*
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1æ
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3quant_conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿA
8:8:8:8:8:*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_1/ConstÖ
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool¤
quant_conv2d_2/ste_sign_4/SignSign max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82 
quant_conv2d_2/ste_sign_4/Sign
quant_conv2d_2/ste_sign_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2!
quant_conv2d_2/ste_sign_4/add/yÏ
quant_conv2d_2/ste_sign_4/addAddV2"quant_conv2d_2/ste_sign_4/Sign:y:0(quant_conv2d_2/ste_sign_4/add/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82
quant_conv2d_2/ste_sign_4/add©
 quant_conv2d_2/ste_sign_4/Sign_1Sign!quant_conv2d_2/ste_sign_4/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82"
 quant_conv2d_2/ste_sign_4/Sign_1´
"quant_conv2d_2/ste_sign_4/IdentityIdentity$quant_conv2d_2/ste_sign_4/Sign_1:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82$
"quant_conv2d_2/ste_sign_4/Identity¦
#quant_conv2d_2/ste_sign_4/IdentityN	IdentityN$quant_conv2d_2/ste_sign_4/Sign_1:y:0 max_pooling2d_1/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-183884*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ 8:ÿÿÿÿÿÿÿÿÿ 82%
#quant_conv2d_2/ste_sign_4/IdentityNÂ
$quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_2/Conv2D/ReadVariableOpµ
%quant_conv2d_2/Conv2D/ste_sign_3/SignSign,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:882'
%quant_conv2d_2/Conv2D/ste_sign_3/Sign
&quant_conv2d_2/Conv2D/ste_sign_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2(
&quant_conv2d_2/Conv2D/ste_sign_3/add/yâ
$quant_conv2d_2/Conv2D/ste_sign_3/addAddV2)quant_conv2d_2/Conv2D/ste_sign_3/Sign:y:0/quant_conv2d_2/Conv2D/ste_sign_3/add/y:output:0*
T0*&
_output_shapes
:882&
$quant_conv2d_2/Conv2D/ste_sign_3/addµ
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1Sign(quant_conv2d_2/Conv2D/ste_sign_3/add:z:0*
T0*&
_output_shapes
:882)
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1À
)quant_conv2d_2/Conv2D/ste_sign_3/IdentityIdentity+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0*
T0*&
_output_shapes
:882+
)quant_conv2d_2/Conv2D/ste_sign_3/Identityµ
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityN	IdentityN+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183894*8
_output_shapes&
$:88:882,
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityNý
quant_conv2d_2/Conv2DConv2D,quant_conv2d_2/ste_sign_4/IdentityN:output:03quant_conv2d_2/Conv2D/ste_sign_3/IdentityN:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*
paddingSAME*
strides
2
quant_conv2d_2/Conv2D
"batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_2/LogicalAnd/x
"batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/yÄ
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAnd¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:8*
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:8*
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1æ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3quant_conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ 8:8:8:8:8:*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_2/ConstÖ
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape
quant_dense/ste_sign_6/SignSignflatten/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense/ste_sign_6/Sign
quant_dense/ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
quant_dense/ste_sign_6/add/y¼
quant_dense/ste_sign_6/addAddV2quant_dense/ste_sign_6/Sign:y:0%quant_dense/ste_sign_6/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense/ste_sign_6/add
quant_dense/ste_sign_6/Sign_1Signquant_dense/ste_sign_6/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense/ste_sign_6/Sign_1¤
quant_dense/ste_sign_6/IdentityIdentity!quant_dense/ste_sign_6/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
quant_dense/ste_sign_6/Identity
 quant_dense/ste_sign_6/IdentityN	IdentityN!quant_dense/ste_sign_6/Sign_1:y:0flatten/Reshape:output:0*
T
2*,
_gradient_op_typeCustomGradient-183924*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2"
 quant_dense/ste_sign_6/IdentityN³
!quant_dense/MatMul/ReadVariableOpReadVariableOp*quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!quant_dense/MatMul/ReadVariableOp¦
"quant_dense/MatMul/ste_sign_5/SignSign)quant_dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"quant_dense/MatMul/ste_sign_5/Sign
#quant_dense/MatMul/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2%
#quant_dense/MatMul/ste_sign_5/add/yÐ
!quant_dense/MatMul/ste_sign_5/addAddV2&quant_dense/MatMul/ste_sign_5/Sign:y:0,quant_dense/MatMul/ste_sign_5/add/y:output:0*
T0* 
_output_shapes
:
2#
!quant_dense/MatMul/ste_sign_5/add¦
$quant_dense/MatMul/ste_sign_5/Sign_1Sign%quant_dense/MatMul/ste_sign_5/add:z:0*
T0* 
_output_shapes
:
2&
$quant_dense/MatMul/ste_sign_5/Sign_1±
&quant_dense/MatMul/ste_sign_5/IdentityIdentity(quant_dense/MatMul/ste_sign_5/Sign_1:y:0*
T0* 
_output_shapes
:
2(
&quant_dense/MatMul/ste_sign_5/Identity
'quant_dense/MatMul/ste_sign_5/IdentityN	IdentityN(quant_dense/MatMul/ste_sign_5/Sign_1:y:0)quant_dense/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183934*,
_output_shapes
:
:
2)
'quant_dense/MatMul/ste_sign_5/IdentityNÂ
quant_dense/MatMulMatMul)quant_dense/ste_sign_6/IdentityN:output:00quant_dense/MatMul/ste_sign_5/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense/MatMul
"batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_3/LogicalAnd/x
"batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/yÄ
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAndÕ
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_3/batchnorm/add/yá
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/add¦
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_3/batchnorm/Rsqrtá
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/mulÏ
%batch_normalization_3/batchnorm/mul_1Mulquant_dense/MatMul:product:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/mul_1Û
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Þ
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_3/batchnorm/mul_2Û
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2Ü
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/subÞ
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_3/batchnorm/add_1¤
quant_dense_1/ste_sign_8/SignSign)batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense_1/ste_sign_8/Sign
quant_dense_1/ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2 
quant_dense_1/ste_sign_8/add/yÄ
quant_dense_1/ste_sign_8/addAddV2!quant_dense_1/ste_sign_8/Sign:y:0'quant_dense_1/ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense_1/ste_sign_8/add
quant_dense_1/ste_sign_8/Sign_1Sign quant_dense_1/ste_sign_8/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
quant_dense_1/ste_sign_8/Sign_1ª
!quant_dense_1/ste_sign_8/IdentityIdentity#quant_dense_1/ste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!quant_dense_1/ste_sign_8/Identity
"quant_dense_1/ste_sign_8/IdentityN	IdentityN#quant_dense_1/ste_sign_8/Sign_1:y:0)batch_normalization_3/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-183962*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2$
"quant_dense_1/ste_sign_8/IdentityN¹
#quant_dense_1/MatMul/ReadVariableOpReadVariableOp,quant_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#quant_dense_1/MatMul/ReadVariableOp¬
$quant_dense_1/MatMul/ste_sign_7/SignSign+quant_dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$quant_dense_1/MatMul/ste_sign_7/Sign
%quant_dense_1/MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2'
%quant_dense_1/MatMul/ste_sign_7/add/yØ
#quant_dense_1/MatMul/ste_sign_7/addAddV2(quant_dense_1/MatMul/ste_sign_7/Sign:y:0.quant_dense_1/MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
2%
#quant_dense_1/MatMul/ste_sign_7/add¬
&quant_dense_1/MatMul/ste_sign_7/Sign_1Sign'quant_dense_1/MatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
2(
&quant_dense_1/MatMul/ste_sign_7/Sign_1·
(quant_dense_1/MatMul/ste_sign_7/IdentityIdentity*quant_dense_1/MatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
2*
(quant_dense_1/MatMul/ste_sign_7/Identity¥
)quant_dense_1/MatMul/ste_sign_7/IdentityN	IdentityN*quant_dense_1/MatMul/ste_sign_7/Sign_1:y:0+quant_dense_1/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-183972*,
_output_shapes
:
:
2+
)quant_dense_1/MatMul/ste_sign_7/IdentityNÊ
quant_dense_1/MatMulMatMul+quant_dense_1/ste_sign_8/IdentityN:output:02quant_dense_1/MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense_1/MatMul
"batch_normalization_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_4/LogicalAnd/x
"batch_normalization_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_4/LogicalAnd/yÄ
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_4/LogicalAndÕ
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_4/batchnorm/add/yá
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/add¦
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_4/batchnorm/Rsqrtá
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpÞ
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/mulÑ
%batch_normalization_4/batchnorm/mul_1Mulquant_dense_1/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_4/batchnorm/mul_1Û
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1Þ
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_4/batchnorm/mul_2Û
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2Ü
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/subÞ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_4/batchnorm/add_1¦
quant_dense_2/ste_sign_10/SignSign)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
quant_dense_2/ste_sign_10/Sign
quant_dense_2/ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2!
quant_dense_2/ste_sign_10/add/yÈ
quant_dense_2/ste_sign_10/addAddV2"quant_dense_2/ste_sign_10/Sign:y:0(quant_dense_2/ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense_2/ste_sign_10/add¢
 quant_dense_2/ste_sign_10/Sign_1Sign!quant_dense_2/ste_sign_10/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 quant_dense_2/ste_sign_10/Sign_1­
"quant_dense_2/ste_sign_10/IdentityIdentity$quant_dense_2/ste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"quant_dense_2/ste_sign_10/Identity¡
#quant_dense_2/ste_sign_10/IdentityN	IdentityN$quant_dense_2/ste_sign_10/Sign_1:y:0)batch_normalization_4/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-184000*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2%
#quant_dense_2/ste_sign_10/IdentityN¸
#quant_dense_2/MatMul/ReadVariableOpReadVariableOp,quant_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#quant_dense_2/MatMul/ReadVariableOp«
$quant_dense_2/MatMul/ste_sign_9/SignSign+quant_dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$quant_dense_2/MatMul/ste_sign_9/Sign
%quant_dense_2/MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2'
%quant_dense_2/MatMul/ste_sign_9/add/y×
#quant_dense_2/MatMul/ste_sign_9/addAddV2(quant_dense_2/MatMul/ste_sign_9/Sign:y:0.quant_dense_2/MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	2%
#quant_dense_2/MatMul/ste_sign_9/add«
&quant_dense_2/MatMul/ste_sign_9/Sign_1Sign'quant_dense_2/MatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	2(
&quant_dense_2/MatMul/ste_sign_9/Sign_1¶
(quant_dense_2/MatMul/ste_sign_9/IdentityIdentity*quant_dense_2/MatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	2*
(quant_dense_2/MatMul/ste_sign_9/Identity£
)quant_dense_2/MatMul/ste_sign_9/IdentityN	IdentityN*quant_dense_2/MatMul/ste_sign_9/Sign_1:y:0+quant_dense_2/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-184010**
_output_shapes
:	:	2+
)quant_dense_2/MatMul/ste_sign_9/IdentityNÊ
quant_dense_2/MatMulMatMul,quant_dense_2/ste_sign_10/IdentityN:output:02quant_dense_2/MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quant_dense_2/MatMul
"batch_normalization_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_5/LogicalAnd/x
"batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_5/LogicalAnd/yÄ
 batch_normalization_5/LogicalAnd
LogicalAnd+batch_normalization_5/LogicalAnd/x:output:0+batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_5/LogicalAndÔ
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_5/batchnorm/add/yà
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/add¥
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtà
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mulÐ
%batch_normalization_5/batchnorm/mul_1Mulquant_dense_2/MatMul:product:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_5/batchnorm/mul_1Ú
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1Ý
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2Ú
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2Û
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/subÝ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_5/batchnorm/add_1
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation/Softmaxõ
IdentityIdentityactivation/Softmax:softmax:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp#^quant_conv2d/Conv2D/ReadVariableOp%^quant_conv2d_1/Conv2D/ReadVariableOp%^quant_conv2d_2/Conv2D/ReadVariableOp"^quant_dense/MatMul/ReadVariableOp$^quant_dense_1/MatMul/ReadVariableOp$^quant_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2d
0batch_normalization_5/batchnorm/ReadVariableOp_10batch_normalization_5/batchnorm/ReadVariableOp_12d
0batch_normalization_5/batchnorm/ReadVariableOp_20batch_normalization_5/batchnorm/ReadVariableOp_22h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2H
"quant_conv2d/Conv2D/ReadVariableOp"quant_conv2d/Conv2D/ReadVariableOp2L
$quant_conv2d_1/Conv2D/ReadVariableOp$quant_conv2d_1/Conv2D/ReadVariableOp2L
$quant_conv2d_2/Conv2D/ReadVariableOp$quant_conv2d_2/Conv2D/ReadVariableOp2F
!quant_dense/MatMul/ReadVariableOp!quant_dense/MatMul/ReadVariableOp2J
#quant_dense_1/MatMul/ReadVariableOp#quant_dense_1/MatMul/ReadVariableOp2J
#quant_dense_2/MatMul/ReadVariableOp#quant_dense_2/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¾
G
+__inference_ste_sign_2_layer_call_fn_185232

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_1815222
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
¼
ô
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_181903

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Å	
d
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_185261

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
add/yy
addAddV2Sign:y:0add/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identityâ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-185252*n
_output_shapes\
Z:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
	IdentityN

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs

©
6__inference_batch_normalization_2_layer_call_fn_184685

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1826122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ 8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
»
_
C__inference_flatten_layer_call_and_return_conditional_losses_184704

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs

£
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_181346

inputs"
conv2d_readvariableop_resource
identity¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02
Conv2D/ReadVariableOp·
Conv2D/ste_sign/PartitionedCallPartitionedCallConv2D/ReadVariableOp:value:0*
Tin
2*
Tout
2*&
_output_shapes
:8*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_ste_sign_layer_call_and_return_conditional_losses_1813352!
Conv2D/ste_sign/PartitionedCall
Conv2D/ste_sign/IdentityIdentity(Conv2D/ste_sign/PartitionedCall:output:0*
T0*&
_output_shapes
:82
Conv2D/ste_sign/Identity¹
Conv2DConv2Dinputs!Conv2D/ste_sign/Identity:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*
paddingSAME*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Õ
G
+__inference_activation_layer_call_fn_185181

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1829002
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
©
6__inference_batch_normalization_3_layer_call_fn_184863

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1820672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Õ0
È
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_184814

inputs
assignmovingavg_184789
assignmovingavg_1_184795)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/184789*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_184789*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpÄ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/184789*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/184789*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_184789AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/184789*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/184795*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_184795*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÎ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184795*
_output_shapes	
:2
AssignMovingAvg_1/subÅ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184795*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_184795AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/184795*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1´
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ö
ò
O__inference_batch_normalization_layer_call_and_return_conditional_losses_184320

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ë
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstÛ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ã
Ã
+__inference_sequential_layer_call_fn_183137
quant_conv2d_input
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

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1830742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*©
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_namequant_conv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
á
D
(__inference_flatten_layer_call_fn_184709

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1826772
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
»
_
C__inference_flatten_layer_call_and_return_conditional_losses_182677

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
 
r
,__inference_quant_dense_layer_call_fn_184739

inputs
unknown
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_1827082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ì
L
0__inference_max_pooling2d_2_layer_call_fn_181926

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1819202
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
b
D__inference_ste_sign_layer_call_and_return_conditional_losses_181335

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:82
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:82
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:82
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:82

Identity¬
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-181326*8
_output_shapes&
$:8:82
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:82

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:8:N J
&
_output_shapes
:8
 
_user_specified_nameinputs
¢
t
.__inference_quant_dense_2_layer_call_fn_185047

inputs
unknown
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1828482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
è
©
6__inference_batch_normalization_5_layer_call_fn_185158

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1823352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¯%

O__inference_batch_normalization_layer_call_and_return_conditional_losses_182422

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_182407
assignmovingavg_1_182414
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/182407*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x¯
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/182407*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_182407*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpÌ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/182407*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/182407*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_182407AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/182407*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/182414*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182414*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_182414*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpØ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182414*
_output_shapes
:82
AssignMovingAvg_1/sub_1¿
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/182414*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_182414AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/182414*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp§
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ô
ô
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_182634

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ 8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstÚ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ 8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Å	
d
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_181522

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
add/yy
addAddV2Sign:y:0add/y:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identityâ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-181513*n
_output_shapes\
Z:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
	IdentityN

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
¼
ô
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_184414

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constì
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ì
serving_default¸
Z
quant_conv2d_inputD
$serving_default_quant_conv2d_input:0ÿÿÿÿÿÿÿÿÿ>

activation0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ôò

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
ò__call__
+ó&call_and_return_all_conditional_losses
ô_default_save_signature"¦
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "QuantConv2D", "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "QuantDense", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}], "build_input_shape": [null, 130, 20, 1]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "QuantConv2D", "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "QuantDense", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}], "build_input_shape": [null, 130, 20, 1]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": ["accuracy", "sparse_categorical_accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0013894954463467002, "decay": 0.0, "beta_1": 0.949999988079071, "beta_2": 0.9990000128746033, "epsilon": 1e-08, "amsgrad": false}}}}
«

kernel_quantizer

kernel
trainable_variables
	variables
regularization_losses
	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "QuantConv2D", "name": "quant_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 130, 20, 1], "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
°
axis
	gamma
 beta
!moving_mean
"moving_variance
#trainable_variables
$	variables
%regularization_losses
&	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"Ú
_tf_keras_layerÀ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 56}}}}
û
'trainable_variables
(	variables
)regularization_losses
*	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"ê
_tf_keras_layerÐ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

+kernel_quantizer
,input_quantizer

-kernel
.trainable_variables
/	variables
0regularization_losses
1	keras_api
û__call__
+ü&call_and_return_all_conditional_losses"¹	
_tf_keras_layer	{"class_name": "QuantConv2D", "name": "quant_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 56}}}}
´
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7trainable_variables
8	variables
9regularization_losses
:	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 56}}}}
ÿ
;trainable_variables
<	variables
=regularization_losses
>	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

?kernel_quantizer
@input_quantizer

Akernel
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
__call__
+&call_and_return_all_conditional_losses"¹	
_tf_keras_layer	{"class_name": "QuantConv2D", "name": "quant_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 56}}}}
´
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
__call__
+&call_and_return_all_conditional_losses"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 56}}}}
ÿ
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
__call__
+&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
®
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ò	
Wkernel_quantizer
Xinput_quantizer

Ykernel
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
__call__
+&call_and_return_all_conditional_losses"ª
_tf_keras_layer{"class_name": "QuantDense", "name": "quant_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1792}}}}
µ
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
__call__
+&call_and_return_all_conditional_losses"ß
_tf_keras_layerÅ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 384}}}}
õ	
gkernel_quantizer
hinput_quantizer

ikernel
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
__call__
+&call_and_return_all_conditional_losses"­
_tf_keras_layer{"class_name": "QuantDense", "name": "quant_dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}}
µ
naxis
	ogamma
pbeta
qmoving_mean
rmoving_variance
strainable_variables
t	variables
uregularization_losses
v	keras_api
__call__
+&call_and_return_all_conditional_losses"ß
_tf_keras_layerÅ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 384}}}}
ô	
wkernel_quantizer
xinput_quantizer

ykernel
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
__call__
+&call_and_return_all_conditional_losses"¬
_tf_keras_layer{"class_name": "QuantDense", "name": "quant_dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}}
º
~axis
	gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ý
_tf_keras_layerÃ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 8}}}}
¤
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerõ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}
Â
	iter
beta_1
beta_2

decay
learning_ratemÎmÏ mÐ-mÑ3mÒ4mÓAmÔGmÕHmÖYm×_mØ`mÙimÚomÛpmÜymÝmÞ	mßvàvá vâ-vã3vä4våAvæGvçHvèYvé_vê`vëivìovípvîyvïvð	vñ"
	optimizer
§
0
1
 2
-3
34
45
A6
G7
H8
Y9
_10
`11
i12
o13
p14
y15
16
17"
trackable_list_wrapper

0
1
 2
!3
"4
-5
36
47
58
69
A10
G11
H12
I13
J14
Y15
_16
`17
a18
b19
i20
o21
p22
q23
r24
y25
26
27
28
29"
trackable_list_wrapper
 "
trackable_list_wrapper
¿
 layer_regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
	variables
regularization_losses
ò__call__
ô_default_save_signature
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
­
_custom_metrics
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerè{"class_name": "SteSign", "name": "ste_sign", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
-:+82quant_conv2d/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 layer_regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
	variables
regularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%82batch_normalization/gamma
&:$82batch_normalization/beta
/:-8 (2batch_normalization/moving_mean
3:18 (2#batch_normalization/moving_variance
.
0
 1"
trackable_list_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 layer_regularization_losses
#trainable_variables
metrics
non_trainable_variables
 layers
$	variables
%regularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 ¡layer_regularization_losses
'trainable_variables
¢metrics
£non_trainable_variables
¤layers
(	variables
)regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
±
¥_custom_metrics
¦trainable_variables
§	variables
¨regularization_losses
©	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerì{"class_name": "SteSign", "name": "ste_sign_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}

ªtrainable_variables
«	variables
¬regularization_losses
­	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerì{"class_name": "SteSign", "name": "ste_sign_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
/:-882quant_conv2d_1/kernel
'
-0"
trackable_list_wrapper
'
-0"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 ®layer_regularization_losses
.trainable_variables
¯metrics
°non_trainable_variables
±layers
/	variables
0regularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'82batch_normalization_1/gamma
(:&82batch_normalization_1/beta
1:/8 (2!batch_normalization_1/moving_mean
5:38 (2%batch_normalization_1/moving_variance
.
30
41"
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 ²layer_regularization_losses
7trainable_variables
³metrics
´non_trainable_variables
µlayers
8	variables
9regularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 ¶layer_regularization_losses
;trainable_variables
·metrics
¸non_trainable_variables
¹layers
<	variables
=regularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
±
º_custom_metrics
»trainable_variables
¼	variables
½regularization_losses
¾	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerì{"class_name": "SteSign", "name": "ste_sign_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}

¿trainable_variables
À	variables
Áregularization_losses
Â	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"
_tf_keras_layerì{"class_name": "SteSign", "name": "ste_sign_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
/:-882quant_conv2d_2/kernel
'
A0"
trackable_list_wrapper
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 Ãlayer_regularization_losses
Btrainable_variables
Ämetrics
Ånon_trainable_variables
Ælayers
C	variables
Dregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'82batch_normalization_2/gamma
(:&82batch_normalization_2/beta
1:/8 (2!batch_normalization_2/moving_mean
5:38 (2%batch_normalization_2/moving_variance
.
G0
H1"
trackable_list_wrapper
<
G0
H1
I2
J3"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 Çlayer_regularization_losses
Ktrainable_variables
Èmetrics
Énon_trainable_variables
Êlayers
L	variables
Mregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 Ëlayer_regularization_losses
Otrainable_variables
Ìmetrics
Ínon_trainable_variables
Îlayers
P	variables
Qregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 Ïlayer_regularization_losses
Strainable_variables
Ðmetrics
Ñnon_trainable_variables
Òlayers
T	variables
Uregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
±
Ó_custom_metrics
Ôtrainable_variables
Õ	variables
Öregularization_losses
×	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"
_tf_keras_layerì{"class_name": "SteSign", "name": "ste_sign_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}

Øtrainable_variables
Ù	variables
Úregularization_losses
Û	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"
_tf_keras_layerì{"class_name": "SteSign", "name": "ste_sign_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
&:$
2quant_dense/kernel
'
Y0"
trackable_list_wrapper
'
Y0"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 Ülayer_regularization_losses
Ztrainable_variables
Ýmetrics
Þnon_trainable_variables
ßlayers
[	variables
\regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_3/gamma
):'2batch_normalization_3/beta
2:0 (2!batch_normalization_3/moving_mean
6:4 (2%batch_normalization_3/moving_variance
.
_0
`1"
trackable_list_wrapper
<
_0
`1
a2
b3"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 àlayer_regularization_losses
ctrainable_variables
ámetrics
ânon_trainable_variables
ãlayers
d	variables
eregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
±
ä_custom_metrics
åtrainable_variables
æ	variables
çregularization_losses
è	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layerì{"class_name": "SteSign", "name": "ste_sign_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}

étrainable_variables
ê	variables
ëregularization_losses
ì	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layerì{"class_name": "SteSign", "name": "ste_sign_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
(:&
2quant_dense_1/kernel
'
i0"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 ílayer_regularization_losses
jtrainable_variables
îmetrics
ïnon_trainable_variables
ðlayers
k	variables
lregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_4/gamma
):'2batch_normalization_4/beta
2:0 (2!batch_normalization_4/moving_mean
6:4 (2%batch_normalization_4/moving_variance
.
o0
p1"
trackable_list_wrapper
<
o0
p1
q2
r3"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 ñlayer_regularization_losses
strainable_variables
òmetrics
ónon_trainable_variables
ôlayers
t	variables
uregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
±
õ_custom_metrics
ötrainable_variables
÷	variables
øregularization_losses
ù	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layerì{"class_name": "SteSign", "name": "ste_sign_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}

útrainable_variables
û	variables
üregularization_losses
ý	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layerî{"class_name": "SteSign", "name": "ste_sign_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
':%	2quant_dense_2/kernel
'
y0"
trackable_list_wrapper
'
y0"
trackable_list_wrapper
 "
trackable_list_wrapper
¡
 þlayer_regularization_losses
ztrainable_variables
ÿmetrics
non_trainable_variables
layers
{	variables
|regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
/
0
1"
trackable_list_wrapper
?
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 layer_regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
	variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 layer_regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
	variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
x
!0
"1
52
63
I4
J5
a6
b7
q8
r9
10
11"
trackable_list_wrapper

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
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 layer_regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
	variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 layer_regularization_losses
¦trainable_variables
metrics
non_trainable_variables
layers
§	variables
¨regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 layer_regularization_losses
ªtrainable_variables
metrics
non_trainable_variables
layers
«	variables
¬regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 layer_regularization_losses
»trainable_variables
metrics
non_trainable_variables
layers
¼	variables
½regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 layer_regularization_losses
¿trainable_variables
metrics
non_trainable_variables
layers
À	variables
Áregularization_losses
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
  layer_regularization_losses
Ôtrainable_variables
¡metrics
¢non_trainable_variables
£layers
Õ	variables
Öregularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 ¤layer_regularization_losses
Øtrainable_variables
¥metrics
¦non_trainable_variables
§layers
Ù	variables
Úregularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 ¨layer_regularization_losses
åtrainable_variables
©metrics
ªnon_trainable_variables
«layers
æ	variables
çregularization_losses
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 ¬layer_regularization_losses
étrainable_variables
­metrics
®non_trainable_variables
¯layers
ê	variables
ëregularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 °layer_regularization_losses
ötrainable_variables
±metrics
²non_trainable_variables
³layers
÷	variables
øregularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 ´layer_regularization_losses
útrainable_variables
µmetrics
¶non_trainable_variables
·layers
û	variables
üregularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
£

¸total

¹count
º
_fn_kwargs
»trainable_variables
¼	variables
½regularization_losses
¾	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"å
_tf_keras_layerË{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
É

¿total

Àcount
Á
_fn_kwargs
Âtrainable_variables
Ã	variables
Äregularization_losses
Å	keras_api
°__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layerñ{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
¸0
¹1"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 Ælayer_regularization_losses
»trainable_variables
Çmetrics
Ènon_trainable_variables
Élayers
¼	variables
½regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
¿0
À1"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
 Êlayer_regularization_losses
Âtrainable_variables
Ëmetrics
Ìnon_trainable_variables
Ílayers
Ã	variables
Äregularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
¸0
¹1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
¿0
À1"
trackable_list_wrapper
 "
trackable_list_wrapper
2:082Adam/quant_conv2d/kernel/m
,:*82 Adam/batch_normalization/gamma/m
+:)82Adam/batch_normalization/beta/m
4:2882Adam/quant_conv2d_1/kernel/m
.:,82"Adam/batch_normalization_1/gamma/m
-:+82!Adam/batch_normalization_1/beta/m
4:2882Adam/quant_conv2d_2/kernel/m
.:,82"Adam/batch_normalization_2/gamma/m
-:+82!Adam/batch_normalization_2/beta/m
+:)
2Adam/quant_dense/kernel/m
/:-2"Adam/batch_normalization_3/gamma/m
.:,2!Adam/batch_normalization_3/beta/m
-:+
2Adam/quant_dense_1/kernel/m
/:-2"Adam/batch_normalization_4/gamma/m
.:,2!Adam/batch_normalization_4/beta/m
,:*	2Adam/quant_dense_2/kernel/m
.:,2"Adam/batch_normalization_5/gamma/m
-:+2!Adam/batch_normalization_5/beta/m
2:082Adam/quant_conv2d/kernel/v
,:*82 Adam/batch_normalization/gamma/v
+:)82Adam/batch_normalization/beta/v
4:2882Adam/quant_conv2d_1/kernel/v
.:,82"Adam/batch_normalization_1/gamma/v
-:+82!Adam/batch_normalization_1/beta/v
4:2882Adam/quant_conv2d_2/kernel/v
.:,82"Adam/batch_normalization_2/gamma/v
-:+82!Adam/batch_normalization_2/beta/v
+:)
2Adam/quant_dense/kernel/v
/:-2"Adam/batch_normalization_3/gamma/v
.:,2!Adam/batch_normalization_3/beta/v
-:+
2Adam/quant_dense_1/kernel/v
/:-2"Adam/batch_normalization_4/gamma/v
.:,2!Adam/batch_normalization_4/beta/v
,:*	2Adam/quant_dense_2/kernel/v
.:,2"Adam/batch_normalization_5/gamma/v
-:+2!Adam/batch_normalization_5/beta/v
ú2÷
+__inference_sequential_layer_call_fn_184105
+__inference_sequential_layer_call_fn_183137
+__inference_sequential_layer_call_fn_183283
+__inference_sequential_layer_call_fn_184170À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_183813
F__inference_sequential_layer_call_and_return_conditional_losses_182990
F__inference_sequential_layer_call_and_return_conditional_losses_184040
F__inference_sequential_layer_call_and_return_conditional_losses_182909À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
!__inference__wrapped_model_181317Ê
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *:¢7
52
quant_conv2d_inputÿÿÿÿÿÿÿÿÿ
2
-__inference_quant_conv2d_layer_call_fn_181354×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
§2¤
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_181346×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
4__inference_batch_normalization_layer_call_fn_184251
4__inference_batch_normalization_layer_call_fn_184333
4__inference_batch_normalization_layer_call_fn_184264
4__inference_batch_normalization_layer_call_fn_184346´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þ2û
O__inference_batch_normalization_layer_call_and_return_conditional_losses_184238
O__inference_batch_normalization_layer_call_and_return_conditional_losses_184216
O__inference_batch_normalization_layer_call_and_return_conditional_losses_184298
O__inference_batch_normalization_layer_call_and_return_conditional_losses_184320´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_max_pooling2d_layer_call_fn_181506à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_181500à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_quant_conv2d_1_layer_call_fn_181564×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
©2¦
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_181556×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
2
6__inference_batch_normalization_1_layer_call_fn_184522
6__inference_batch_normalization_1_layer_call_fn_184509
6__inference_batch_normalization_1_layer_call_fn_184440
6__inference_batch_normalization_1_layer_call_fn_184427´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_184392
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_184474
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_184414
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_184496´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
0__inference_max_pooling2d_1_layer_call_fn_181716à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_181710à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_quant_conv2d_2_layer_call_fn_181774×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
©2¦
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_181766×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
2
6__inference_batch_normalization_2_layer_call_fn_184698
6__inference_batch_normalization_2_layer_call_fn_184616
6__inference_batch_normalization_2_layer_call_fn_184603
6__inference_batch_normalization_2_layer_call_fn_184685´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184590
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184672
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184568
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184650´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
0__inference_max_pooling2d_2_layer_call_fn_181926à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_181920à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ò2Ï
(__inference_flatten_layer_call_fn_184709¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_flatten_layer_call_and_return_conditional_losses_184704¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_quant_dense_layer_call_fn_184739¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_quant_dense_layer_call_and_return_conditional_losses_184732¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª2§
6__inference_batch_normalization_3_layer_call_fn_184850
6__inference_batch_normalization_3_layer_call_fn_184863´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_184814
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_184837´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
.__inference_quant_dense_1_layer_call_fn_184893¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_184886¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª2§
6__inference_batch_normalization_4_layer_call_fn_185004
6__inference_batch_normalization_4_layer_call_fn_185017´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184991
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184968´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
.__inference_quant_dense_2_layer_call_fn_185047¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_185040¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª2§
6__inference_batch_normalization_5_layer_call_fn_185171
6__inference_batch_normalization_5_layer_call_fn_185158´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_185122
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_185145´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_activation_layer_call_fn_185181¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_activation_layer_call_and_return_conditional_losses_185176¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
>B<
$__inference_signature_wrapper_183502quant_conv2d_input
Ó2Ð
)__inference_ste_sign_layer_call_fn_185198¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_ste_sign_layer_call_and_return_conditional_losses_185193¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_ste_sign_1_layer_call_fn_185215¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_185210¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_ste_sign_2_layer_call_fn_185232¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_185227¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_ste_sign_3_layer_call_fn_185249¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_185244¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_ste_sign_4_layer_call_fn_185266¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_185261¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 È
!__inference__wrapped_model_181317¢! !"-3456AGHIJYb_a`iroqpyD¢A
:¢7
52
quant_conv2d_inputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

activation$!

activationÿÿÿÿÿÿÿÿÿ¢
F__inference_activation_layer_call_and_return_conditional_losses_185176X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
+__inference_activation_layer_call_fn_185181K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿì
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1843923456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 ì
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1844143456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 Ç
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_184474r3456;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿA
8
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿA
8
 Ç
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_184496r3456;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿA
8
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿA
8
 Ä
6__inference_batch_normalization_1_layer_call_fn_1844273456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8Ä
6__inference_batch_normalization_1_layer_call_fn_1844403456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
6__inference_batch_normalization_1_layer_call_fn_184509e3456;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿA
8
p
ª " ÿÿÿÿÿÿÿÿÿA
8
6__inference_batch_normalization_1_layer_call_fn_184522e3456;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿA
8
p 
ª " ÿÿÿÿÿÿÿÿÿA
8ì
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184568GHIJM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 ì
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184590GHIJM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 Ç
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184650rGHIJ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 8
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 8
 Ç
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_184672rGHIJ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 8
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 8
 Ä
6__inference_batch_normalization_2_layer_call_fn_184603GHIJM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8Ä
6__inference_batch_normalization_2_layer_call_fn_184616GHIJM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
6__inference_batch_normalization_2_layer_call_fn_184685eGHIJ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 8
p
ª " ÿÿÿÿÿÿÿÿÿ 8
6__inference_batch_normalization_2_layer_call_fn_184698eGHIJ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 8
p 
ª " ÿÿÿÿÿÿÿÿÿ 8¹
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_184814dab_`4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¹
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_184837db_a`4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_3_layer_call_fn_184850Wab_`4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_3_layer_call_fn_184863Wb_a`4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¹
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184968dqrop4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¹
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184991droqp4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_4_layer_call_fn_185004Wqrop4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_4_layer_call_fn_185017Wroqp4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿº
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_185122e3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_185145e3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_5_layer_call_fn_185158X3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_5_layer_call_fn_185171X3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿê
O__inference_batch_normalization_layer_call_and_return_conditional_losses_184216 !"M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 ê
O__inference_batch_normalization_layer_call_and_return_conditional_losses_184238 !"M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 Ç
O__inference_batch_normalization_layer_call_and_return_conditional_losses_184298t !"<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ8
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ8
 Ç
O__inference_batch_normalization_layer_call_and_return_conditional_losses_184320t !"<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ8
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ8
 Â
4__inference_batch_normalization_layer_call_fn_184251 !"M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8Â
4__inference_batch_normalization_layer_call_fn_184264 !"M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
4__inference_batch_normalization_layer_call_fn_184333g !"<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ8
p
ª "!ÿÿÿÿÿÿÿÿÿ8
4__inference_batch_normalization_layer_call_fn_184346g !"<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ8
p 
ª "!ÿÿÿÿÿÿÿÿÿ8¨
C__inference_flatten_layer_call_and_return_conditional_losses_184704a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ8
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_flatten_layer_call_fn_184709T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ8
ª "ÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_181710R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_1_layer_call_fn_181716R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_181920R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_2_layer_call_fn_181926R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_181500R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_layer_call_fn_181506R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÞ
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_181556-I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 ¶
/__inference_quant_conv2d_1_layer_call_fn_181564-I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8Þ
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_181766AI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 ¶
/__inference_quant_conv2d_2_layer_call_fn_181774AI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8Ü
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_181346I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 ´
-__inference_quant_conv2d_layer_call_fn_181354I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8ª
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_184886]i0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_quant_dense_1_layer_call_fn_184893Pi0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_185040\y0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_quant_dense_2_layer_call_fn_185047Oy0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
G__inference_quant_dense_layer_call_and_return_conditional_losses_184732]Y0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_quant_dense_layer_call_fn_184739PY0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿã
F__inference_sequential_layer_call_and_return_conditional_losses_182909! !"-3456AGHIJYab_`iqropyL¢I
B¢?
52
quant_conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ã
F__inference_sequential_layer_call_and_return_conditional_losses_182990! !"-3456AGHIJYb_a`iroqpyL¢I
B¢?
52
quant_conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ×
F__inference_sequential_layer_call_and_return_conditional_losses_183813! !"-3456AGHIJYab_`iqropy@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ×
F__inference_sequential_layer_call_and_return_conditional_losses_184040! !"-3456AGHIJYb_a`iroqpy@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
+__inference_sequential_layer_call_fn_183137! !"-3456AGHIJYab_`iqropyL¢I
B¢?
52
quant_conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ»
+__inference_sequential_layer_call_fn_183283! !"-3456AGHIJYb_a`iroqpyL¢I
B¢?
52
quant_conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ®
+__inference_sequential_layer_call_fn_184105! !"-3456AGHIJYab_`iqropy@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ®
+__inference_sequential_layer_call_fn_184170! !"-3456AGHIJYb_a`iroqpy@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿá
$__inference_signature_wrapper_183502¸! !"-3456AGHIJYb_a`iroqpyZ¢W
¢ 
PªM
K
quant_conv2d_input52
quant_conv2d_inputÿÿÿÿÿÿÿÿÿ"7ª4
2

activation$!

activationÿÿÿÿÿÿÿÿÿ 
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_185210V.¢+
$¢!

inputs88
ª "$¢!

088
 x
+__inference_ste_sign_1_layer_call_fn_185215I.¢+
$¢!

inputs88
ª "88×
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_185227I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 ®
+__inference_ste_sign_2_layer_call_fn_185232I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8 
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_185244V.¢+
$¢!

inputs88
ª "$¢!

088
 x
+__inference_ste_sign_3_layer_call_fn_185249I.¢+
$¢!

inputs88
ª "88×
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_185261I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 ®
+__inference_ste_sign_4_layer_call_fn_185266I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
D__inference_ste_sign_layer_call_and_return_conditional_losses_185193V.¢+
$¢!

inputs8
ª "$¢!

08
 v
)__inference_ste_sign_layer_call_fn_185198I.¢+
$¢!

inputs8
ª "8