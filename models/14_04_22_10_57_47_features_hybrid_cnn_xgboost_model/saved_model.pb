κδ 
«ύ
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
Ύ
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
shapeshape"serve*2.1.02v1.12.1-25073-g2c5e22190c8
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
’
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
’
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
’
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

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Υ
valueΚBΖ BΎ
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
^

kernel
regularization_losses
	variables
trainable_variables
	keras_api

axis
	gamma
beta
 moving_mean
!moving_variance
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
^

*kernel
+regularization_losses
,	variables
-trainable_variables
.	keras_api

/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5	variables
6trainable_variables
7	keras_api
R
8regularization_losses
9	variables
:trainable_variables
;	keras_api
^

<kernel
=regularization_losses
>	variables
?trainable_variables
@	keras_api

Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
R
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
^

Rkernel
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api

Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\regularization_losses
]	variables
^trainable_variables
_	keras_api
^

`kernel
aregularization_losses
b	variables
ctrainable_variables
d	keras_api

eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
^

nkernel
oregularization_losses
p	variables
qtrainable_variables
r	keras_api

saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
R
|regularization_losses
}	variables
~trainable_variables
	keras_api
­
	iter
beta_1
beta_2

decay
learning_ratemεmζmη*mθ0mι1mκ<mλBmμCmνRmξXmοYmπ`mρfmςgmσnmτtmυumφvχvψvω*vϊ0vϋ1vό<vύBvώCv?RvXvYv`vfvgvnvtvuv
 
ζ
0
1
2
 3
!4
*5
06
17
28
39
<10
B11
C12
D13
E14
R15
X16
Y17
Z18
[19
`20
f21
g22
h23
i24
n25
t26
u27
v28
w29

0
1
2
*3
04
15
<6
B7
C8
R9
X10
Y11
`12
f13
g14
n15
t16
u17

regularization_losses
	variables
 layer_regularization_losses
trainable_variables
non_trainable_variables
metrics
layers
 
_]
VARIABLE_VALUEquant_conv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0

regularization_losses
	variables
 layer_regularization_losses
trainable_variables
non_trainable_variables
metrics
layers
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 2
!3

0
1

"regularization_losses
#	variables
 layer_regularization_losses
$trainable_variables
non_trainable_variables
metrics
layers
 
 
 

&regularization_losses
'	variables
 layer_regularization_losses
(trainable_variables
non_trainable_variables
metrics
layers
a_
VARIABLE_VALUEquant_conv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

*0

*0

+regularization_losses
,	variables
 layer_regularization_losses
-trainable_variables
non_trainable_variables
metrics
layers
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

00
11
22
33

00
11

4regularization_losses
5	variables
 layer_regularization_losses
6trainable_variables
non_trainable_variables
metrics
layers
 
 
 

8regularization_losses
9	variables
 layer_regularization_losses
:trainable_variables
non_trainable_variables
metrics
 layers
a_
VARIABLE_VALUEquant_conv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

<0

<0

=regularization_losses
>	variables
 ‘layer_regularization_losses
?trainable_variables
’non_trainable_variables
£metrics
€layers
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1
D2
E3

B0
C1

Fregularization_losses
G	variables
 ₯layer_regularization_losses
Htrainable_variables
¦non_trainable_variables
§metrics
¨layers
 
 
 

Jregularization_losses
K	variables
 ©layer_regularization_losses
Ltrainable_variables
ͺnon_trainable_variables
«metrics
¬layers
 
 
 

Nregularization_losses
O	variables
 ­layer_regularization_losses
Ptrainable_variables
?non_trainable_variables
―metrics
°layers
^\
VARIABLE_VALUEquant_dense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

R0

R0

Sregularization_losses
T	variables
 ±layer_regularization_losses
Utrainable_variables
²non_trainable_variables
³metrics
΄layers
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1
Z2
[3

X0
Y1

\regularization_losses
]	variables
 ΅layer_regularization_losses
^trainable_variables
Άnon_trainable_variables
·metrics
Έlayers
`^
VARIABLE_VALUEquant_dense_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

`0

`0

aregularization_losses
b	variables
 Ήlayer_regularization_losses
ctrainable_variables
Ίnon_trainable_variables
»metrics
Όlayers
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1
h2
i3

f0
g1

jregularization_losses
k	variables
 ½layer_regularization_losses
ltrainable_variables
Ύnon_trainable_variables
Ώmetrics
ΐlayers
a_
VARIABLE_VALUEquant_dense_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

n0

n0

oregularization_losses
p	variables
 Αlayer_regularization_losses
qtrainable_variables
Βnon_trainable_variables
Γmetrics
Δlayers
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1
v2
w3

t0
u1

xregularization_losses
y	variables
 Εlayer_regularization_losses
ztrainable_variables
Ζnon_trainable_variables
Ηmetrics
Θlayers
 
 
 

|regularization_losses
}	variables
 Ιlayer_regularization_losses
~trainable_variables
Κnon_trainable_variables
Λmetrics
Μlayers
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
V
 0
!1
22
33
D4
E5
Z6
[7
h8
i9
v10
w11

Ν0
Ξ1
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
 

 0
!1
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

20
31
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

D0
E1
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

Z0
[1
 
 
 
 
 
 
 

h0
i1
 
 
 
 
 
 
 

v0
w1
 
 
 
 
 
 


Οtotal

Πcount
Ρ
_fn_kwargs
?regularization_losses
Σ	variables
Τtrainable_variables
Υ	keras_api


Φtotal

Χcount
Ψ
_fn_kwargs
Ωregularization_losses
Ϊ	variables
Ϋtrainable_variables
ά	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

Ο0
Π1
 
‘
?regularization_losses
Σ	variables
 έlayer_regularization_losses
Τtrainable_variables
ήnon_trainable_variables
ίmetrics
ΰlayers
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

Φ0
Χ1
 
‘
Ωregularization_losses
Ϊ	variables
 αlayer_regularization_losses
Ϋtrainable_variables
βnon_trainable_variables
γmetrics
δlayers
 

Ο0
Π1
 
 
 

Φ0
Χ1
 
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
:?????????*
dtype0*%
shape:?????????
’	
StatefulPartitionedCallStatefulPartitionedCall"serving_default_quant_conv2d_inputquant_conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancequant_conv2d_1/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancequant_conv2d_2/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancequant_dense/kernel%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betaquant_dense_1/kernel%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betaquant_dense_2/kernel%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/beta**
Tin#
!2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference_signature_wrapper_174600
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'quant_conv2d/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp)quant_conv2d_1/kernel/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp)quant_conv2d_2/kernel/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp&quant_dense/kernel/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp(quant_dense_1/kernel/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp(quant_dense_2/kernel/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/quant_conv2d/kernel/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp0Adam/quant_conv2d_1/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp0Adam/quant_conv2d_2/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp-Adam/quant_dense/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp/Adam/quant_dense_1/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp/Adam/quant_dense_2/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp.Adam/quant_conv2d/kernel/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp0Adam/quant_conv2d_1/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp0Adam/quant_conv2d_2/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp-Adam/quant_dense/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp/Adam/quant_dense_1/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp/Adam/quant_dense_2/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_save_176304
΅
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamequant_conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancequant_conv2d_1/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancequant_conv2d_2/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancequant_dense/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancequant_dense_1/kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancequant_dense_2/kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/quant_conv2d/kernel/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/quant_conv2d_1/kernel/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/quant_conv2d_2/kernel/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/quant_dense/kernel/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/quant_dense_1/kernel/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/quant_dense_2/kernel/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/quant_conv2d/kernel/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/quant_conv2d_1/kernel/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/quant_conv2d_2/kernel/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/quant_dense/kernel/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/quant_dense_1/kernel/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/quant_dense_2/kernel/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/v*W
TinP
N2L*
Tout
2*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__traced_restore_176541ͺ
½0
Θ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_175996

inputs
assignmovingavg_175971
assignmovingavg_1_175977)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp^
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
moments/StopGradient€
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
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
loc:@AssignMovingAvg/175971*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_175971*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/175971*
_output_shapes
:2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/175971*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_175971AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/175971*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/175977*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_175977*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175977*
_output_shapes
:2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175977*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_175977AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/175977*
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
:?????????2
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
:?????????2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
ι
©
6__inference_batch_normalization_4_layer_call_fn_175894

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1733292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
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
ΓY
Α
F__inference_sequential_layer_call_and_return_conditional_losses_174088
quant_conv2d_input
quant_conv2d_174010
batch_normalization_174013
batch_normalization_174015
batch_normalization_174017
batch_normalization_174019
quant_conv2d_1_174023 
batch_normalization_1_174026 
batch_normalization_1_174028 
batch_normalization_1_174030 
batch_normalization_1_174032
quant_conv2d_2_174036 
batch_normalization_2_174039 
batch_normalization_2_174041 
batch_normalization_2_174043 
batch_normalization_2_174045
quant_dense_174050 
batch_normalization_3_174053 
batch_normalization_3_174055 
batch_normalization_3_174057 
batch_normalization_3_174059
quant_dense_1_174062 
batch_normalization_4_174065 
batch_normalization_4_174067 
batch_normalization_4_174069 
batch_normalization_4_174071
quant_dense_2_174074 
batch_normalization_5_174077 
batch_normalization_5_174079 
batch_normalization_5_174081 
batch_normalization_5_174083
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’-batch_normalization_4/StatefulPartitionedCall’-batch_normalization_5/StatefulPartitionedCall’$quant_conv2d/StatefulPartitionedCall’&quant_conv2d_1/StatefulPartitionedCall’&quant_conv2d_2/StatefulPartitionedCall’#quant_dense/StatefulPartitionedCall’%quant_dense_1/StatefulPartitionedCall’%quant_dense_2/StatefulPartitionedCallί
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputquant_conv2d_174010*
Tin
2*
Tout
2*0
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1725762&
$quant_conv2d/StatefulPartitionedCallν
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_174013batch_normalization_174015batch_normalization_174017batch_normalization_174019*
Tin	
2*
Tout
2*0
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1735902-
+batch_normalization/StatefulPartitionedCallΥ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1727302
max_pooling2d/PartitionedCallϊ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_174023*
Tin
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1727442(
&quant_conv2d_1/StatefulPartitionedCallό
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_174026batch_normalization_1_174028batch_normalization_1_174030batch_normalization_1_174032*
Tin	
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1736852/
-batch_normalization_1/StatefulPartitionedCallέ
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1728982!
max_pooling2d_1/PartitionedCallό
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_174036*
Tin
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1729122(
&quant_conv2d_2/StatefulPartitionedCallό
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_174039batch_normalization_2_174041batch_normalization_2_174043batch_normalization_2_174045*
Tin	
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1737802/
-batch_normalization_2/StatefulPartitionedCallέ
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1730662!
max_pooling2d_2/PartitionedCall°
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1738232
flatten/PartitionedCallα
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_174050*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_1738382%
#quant_dense/StatefulPartitionedCallς
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_174053batch_normalization_3_174055batch_normalization_3_174057batch_normalization_3_174059*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1732132/
-batch_normalization_3/StatefulPartitionedCall?
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_174062*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1738922'
%quant_dense_1/StatefulPartitionedCallτ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_174065batch_normalization_4_174067batch_normalization_4_174069batch_normalization_4_174071*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1733652/
-batch_normalization_4/StatefulPartitionedCallώ
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_174074*
Tin
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1739462'
%quant_dense_2/StatefulPartitionedCallσ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_174077batch_normalization_5_174079batch_normalization_5_174081batch_normalization_5_174083*
Tin	
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1735172/
-batch_normalization_5/StatefulPartitionedCallΖ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1739982
activation/PartitionedCall
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::2Z
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
:?????????
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
ϊ

Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_176019

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp^
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
:?????????2
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
:?????????2
batchnorm/add_1Ϋ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
Ν
©
6__inference_batch_normalization_1_layer_call_fn_175431

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallλ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1728462
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
΄
Ό
$__inference_signature_wrapper_174600
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
identity’StatefulPartitionedCallψ
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
:?????????**
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__wrapped_model_1725682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:?????????
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

©
6__inference_batch_normalization_2_layer_call_fn_175620

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1737802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? 8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 8
 
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175572

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_175557
assignmovingavg_1_175564
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
7:????????? 8:8:8:8:8:*
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
loc:@AssignMovingAvg/175557*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/175557*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_175557*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/175557*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/175557*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_175557AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/175557*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/175564*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175564*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_175564*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175564*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175564*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_175564AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/175564*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¦
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? 8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 8
 
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
ι
©
6__inference_batch_normalization_4_layer_call_fn_175907

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1733652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
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
Ϊ
―$
__inference__traced_save_176304
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

identity_1’MergeV2Checkpoints’SaveV2’SaveV2_1₯
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c95914aa69ff4f49aa3aab57d6cd1ede/part2
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
ShardedFilenameΙ)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ϋ(
valueΡ(BΞ(KB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names‘
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value‘BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesχ"
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
ShardedFilename_1’
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
SaveV2_1/shape_and_slicesΟ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1γ
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

identity_1Identity_1:output:0*ζ
_input_shapesΤ
Ρ: :8:8:8:8:8:88:8:8:8:8:88:8:8:8:8:
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
ΰ
Γ
+__inference_sequential_layer_call_fn_174235
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
identity’StatefulPartitionedCall
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
:?????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1741722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:?????????
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
»
_
C__inference_flatten_layer_call_and_return_conditional_losses_175626

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????8:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
Υ0
Θ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_173177

inputs
assignmovingavg_173152
assignmovingavg_1_173158)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp^
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
moments/StopGradient₯
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:?????????2
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
loc:@AssignMovingAvg/173152*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_173152*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpΔ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/173152*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/173152*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_173152AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/173152*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/173158*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_173158*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΞ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173158*
_output_shapes	
:2
AssignMovingAvg_1/subΕ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173158*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_173158AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/173158*
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
:?????????2
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
:?????????2
batchnorm/add_1΄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
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
ΐΘ
έ,
"__inference__traced_restore_176541
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
identity_76’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_69’AssignVariableOp_7’AssignVariableOp_70’AssignVariableOp_71’AssignVariableOp_72’AssignVariableOp_73’AssignVariableOp_74’AssignVariableOp_8’AssignVariableOp_9’	RestoreV2’RestoreV2_1Ο)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ϋ(
valueΡ(BΞ(KB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names§
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value‘BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices₯
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Β
_output_shapes―
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

Identity_1’
AssignVariableOp_1AssignVariableOp,assignvariableop_1_batch_normalization_gammaIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2‘
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

Identity_6€
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

Identity_8ͺ
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10’
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
Identity_13?
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
Identity_18?
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
Identity_20‘
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
Identity_23?
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
Identity_25‘
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
Identity_28?
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
Identity_43―
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_1_gamma_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
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
Identity_46―
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_2_gamma_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47?
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
Identity_49―
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_3_gamma_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50?
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
Identity_52―
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_batch_normalization_4_gamma_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53?
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
Identity_55―
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_5_gamma_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56?
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
Identity_61―
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_1_gamma_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62?
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
Identity_64―
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_2_gamma_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65?
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
Identity_67―
AssignVariableOp_67AssignVariableOp6assignvariableop_67_adam_batch_normalization_3_gamma_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68?
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
Identity_70―
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_4_gamma_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71?
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
Identity_73―
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_batch_normalization_5_gamma_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74?
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
RestoreV2_1/shape_and_slicesΔ
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
NoOpΠ
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75έ
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*Γ
_input_shapes±
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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

€
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_175914

inputs"
matmul_readvariableop_resource
identity’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
Ό
·
+__inference_sequential_layer_call_fn_175092

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
identity’StatefulPartitionedCall
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
:?????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1743182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
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
¬
£
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_172576

inputs"
conv2d_readvariableop_resource
identity’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????8*
paddingSAME*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: 
Ί
ς
O__inference_batch_normalization_layer_call_and_return_conditional_losses_175160

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constμ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
ή
D
(__inference_flatten_layer_call_fn_175631

inputs
identityώ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1738232
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????8:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
Τ
Χ
F__inference_sequential_layer_call_and_return_conditional_losses_174962

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
identity’3batch_normalization/FusedBatchNormV3/ReadVariableOp’5batch_normalization/FusedBatchNormV3/ReadVariableOp_1’"batch_normalization/ReadVariableOp’$batch_normalization/ReadVariableOp_1’5batch_normalization_1/FusedBatchNormV3/ReadVariableOp’7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’$batch_normalization_1/ReadVariableOp’&batch_normalization_1/ReadVariableOp_1’5batch_normalization_2/FusedBatchNormV3/ReadVariableOp’7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1’$batch_normalization_2/ReadVariableOp’&batch_normalization_2/ReadVariableOp_1’.batch_normalization_3/batchnorm/ReadVariableOp’0batch_normalization_3/batchnorm/ReadVariableOp_1’0batch_normalization_3/batchnorm/ReadVariableOp_2’2batch_normalization_3/batchnorm/mul/ReadVariableOp’.batch_normalization_4/batchnorm/ReadVariableOp’0batch_normalization_4/batchnorm/ReadVariableOp_1’0batch_normalization_4/batchnorm/ReadVariableOp_2’2batch_normalization_4/batchnorm/mul/ReadVariableOp’.batch_normalization_5/batchnorm/ReadVariableOp’0batch_normalization_5/batchnorm/ReadVariableOp_1’0batch_normalization_5/batchnorm/ReadVariableOp_2’2batch_normalization_5/batchnorm/mul/ReadVariableOp’"quant_conv2d/Conv2D/ReadVariableOp’$quant_conv2d_1/Conv2D/ReadVariableOp’$quant_conv2d_2/Conv2D/ReadVariableOp’!quant_dense/MatMul/ReadVariableOp’#quant_dense_1/MatMul/ReadVariableOp’#quant_dense_2/MatMul/ReadVariableOpΌ
"quant_conv2d/Conv2D/ReadVariableOpReadVariableOp+quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02$
"quant_conv2d/Conv2D/ReadVariableOpΛ
quant_conv2d/Conv2DConv2Dinputs*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????8*
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
 batch_normalization/LogicalAnd/yΌ
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2 
batch_normalization/LogicalAnd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype02$
"batch_normalization/ReadVariableOpΆ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype02&
$batch_normalization/ReadVariableOp_1γ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpι
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ω
$batch_normalization/FusedBatchNormV3FusedBatchNormV3quant_conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????8:8:8:8:8:*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3{
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization/ConstΠ
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????A
8*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolΒ
$quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_1/Conv2D/ReadVariableOpθ
quant_conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????A
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
"batch_normalization_1/LogicalAnd/yΔ
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_1/LogicalAndΆ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:8*
dtype02&
$batch_normalization_1/ReadVariableOpΌ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:8*
dtype02(
&batch_normalization_1/ReadVariableOp_1ι
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ζ
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3quant_conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????A
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
batch_normalization_1/ConstΦ
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:????????? 8*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolΒ
$quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_2/Conv2D/ReadVariableOpκ
quant_conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 8*
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
"batch_normalization_2/LogicalAnd/yΔ
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAndΆ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:8*
dtype02&
$batch_normalization_2/ReadVariableOpΌ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:8*
dtype02(
&batch_normalization_2/ReadVariableOp_1ι
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ζ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3quant_conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? 8:8:8:8:8:*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_2/ConstΦ
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????8*
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
valueB"????   2
flatten/Const
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
flatten/Reshape³
!quant_dense/MatMul/ReadVariableOpReadVariableOp*quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!quant_dense/MatMul/ReadVariableOpͺ
quant_dense/MatMulMatMulflatten/Reshape:output:0)quant_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
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
"batch_normalization_3/LogicalAnd/yΔ
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAndΥ
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
%batch_normalization_3/batchnorm/add/yα
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/add¦
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_3/batchnorm/Rsqrtα
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpή
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/mulΟ
%batch_normalization_3/batchnorm/mul_1Mulquant_dense/MatMul:product:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:?????????2'
%batch_normalization_3/batchnorm/mul_1Ϋ
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1ή
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_3/batchnorm/mul_2Ϋ
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2ά
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/subή
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????2'
%batch_normalization_3/batchnorm/add_1Ή
#quant_dense_1/MatMul/ReadVariableOpReadVariableOp,quant_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#quant_dense_1/MatMul/ReadVariableOpΑ
quant_dense_1/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0+quant_dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
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
"batch_normalization_4/LogicalAnd/yΔ
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_4/LogicalAndΥ
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
%batch_normalization_4/batchnorm/add/yα
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/add¦
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_4/batchnorm/Rsqrtα
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpή
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/mulΡ
%batch_normalization_4/batchnorm/mul_1Mulquant_dense_1/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:?????????2'
%batch_normalization_4/batchnorm/mul_1Ϋ
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1ή
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_4/batchnorm/mul_2Ϋ
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2ά
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/subή
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????2'
%batch_normalization_4/batchnorm/add_1Έ
#quant_dense_2/MatMul/ReadVariableOpReadVariableOp,quant_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#quant_dense_2/MatMul/ReadVariableOpΐ
quant_dense_2/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0+quant_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
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
"batch_normalization_5/LogicalAnd/yΔ
 batch_normalization_5/LogicalAnd
LogicalAnd+batch_normalization_5/LogicalAnd/x:output:0+batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_5/LogicalAndΤ
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
%batch_normalization_5/batchnorm/add/yΰ
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/add₯
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtΰ
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpέ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mulΠ
%batch_normalization_5/batchnorm/mul_1Mulquant_dense_2/MatMul:product:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_5/batchnorm/mul_1Ϊ
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1έ
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2Ϊ
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2Ϋ
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/subέ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_5/batchnorm/add_1
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
activation/Softmaxυ
IdentityIdentityactivation/Softmax:softmax:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp#^quant_conv2d/Conv2D/ReadVariableOp%^quant_conv2d_1/Conv2D/ReadVariableOp%^quant_conv2d_2/Conv2D/ReadVariableOp"^quant_dense/MatMul/ReadVariableOp$^quant_dense_1/MatMul/ReadVariableOp$^quant_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::2j
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
:?????????
 
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

©
6__inference_batch_normalization_1_layer_call_fn_175349

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1736632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????A
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????A
8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????A
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

u
/__inference_quant_conv2d_1_layer_call_fn_172752

inputs
unknown
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????8**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1727442
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????8:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????8
 
_user_specified_nameinputs:

_output_shapes
: 

€
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_173892

inputs"
matmul_readvariableop_resource
identity’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
Ν
©
6__inference_batch_normalization_2_layer_call_fn_175538

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallλ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1730492
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_175881

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp^
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
:?????????2
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
:?????????2
batchnorm/add_1ά
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
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
Ό
τ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_173049

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constμ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
τ
τ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175594

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? 8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstΪ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? 8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 8
 
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

€
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_175776

inputs"
matmul_readvariableop_resource
identity’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
Y
΅
F__inference_sequential_layer_call_and_return_conditional_losses_174318

inputs
quant_conv2d_174240
batch_normalization_174243
batch_normalization_174245
batch_normalization_174247
batch_normalization_174249
quant_conv2d_1_174253 
batch_normalization_1_174256 
batch_normalization_1_174258 
batch_normalization_1_174260 
batch_normalization_1_174262
quant_conv2d_2_174266 
batch_normalization_2_174269 
batch_normalization_2_174271 
batch_normalization_2_174273 
batch_normalization_2_174275
quant_dense_174280 
batch_normalization_3_174283 
batch_normalization_3_174285 
batch_normalization_3_174287 
batch_normalization_3_174289
quant_dense_1_174292 
batch_normalization_4_174295 
batch_normalization_4_174297 
batch_normalization_4_174299 
batch_normalization_4_174301
quant_dense_2_174304 
batch_normalization_5_174307 
batch_normalization_5_174309 
batch_normalization_5_174311 
batch_normalization_5_174313
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’-batch_normalization_4/StatefulPartitionedCall’-batch_normalization_5/StatefulPartitionedCall’$quant_conv2d/StatefulPartitionedCall’&quant_conv2d_1/StatefulPartitionedCall’&quant_conv2d_2/StatefulPartitionedCall’#quant_dense/StatefulPartitionedCall’%quant_dense_1/StatefulPartitionedCall’%quant_dense_2/StatefulPartitionedCallΣ
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsquant_conv2d_174240*
Tin
2*
Tout
2*0
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1725762&
$quant_conv2d/StatefulPartitionedCallν
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_174243batch_normalization_174245batch_normalization_174247batch_normalization_174249*
Tin	
2*
Tout
2*0
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1735902-
+batch_normalization/StatefulPartitionedCallΥ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1727302
max_pooling2d/PartitionedCallϊ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_174253*
Tin
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1727442(
&quant_conv2d_1/StatefulPartitionedCallό
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_174256batch_normalization_1_174258batch_normalization_1_174260batch_normalization_1_174262*
Tin	
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1736852/
-batch_normalization_1/StatefulPartitionedCallέ
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1728982!
max_pooling2d_1/PartitionedCallό
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_174266*
Tin
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1729122(
&quant_conv2d_2/StatefulPartitionedCallό
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_174269batch_normalization_2_174271batch_normalization_2_174273batch_normalization_2_174275*
Tin	
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1737802/
-batch_normalization_2/StatefulPartitionedCallέ
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1730662!
max_pooling2d_2/PartitionedCall°
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1738232
flatten/PartitionedCallα
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_174280*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_1738382%
#quant_dense/StatefulPartitionedCallς
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_174283batch_normalization_3_174285batch_normalization_3_174287batch_normalization_3_174289*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1732132/
-batch_normalization_3/StatefulPartitionedCall?
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_174292*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1738922'
%quant_dense_1/StatefulPartitionedCallτ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_174295batch_normalization_4_174297batch_normalization_4_174299batch_normalization_4_174301*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1733652/
-batch_normalization_4/StatefulPartitionedCallώ
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_174304*
Tin
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1739462'
%quant_dense_2/StatefulPartitionedCallσ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_174307batch_normalization_5_174309batch_normalization_5_174311batch_normalization_5_174313*
Tin	
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1735172/
-batch_normalization_5/StatefulPartitionedCallΖ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1739982
activation/PartitionedCall
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::2Z
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
:?????????
 
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
Ν
©
6__inference_batch_normalization_2_layer_call_fn_175525

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallλ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1730142
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
Ι
§
4__inference_batch_normalization_layer_call_fn_175186

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallι
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????8**
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1727132
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_173663

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_173648
assignmovingavg_1_173655
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
7:?????????A
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
loc:@AssignMovingAvg/173648*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/173648*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_173648*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/173648*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/173648*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_173648AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/173648*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/173655*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173655*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_173655*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173655*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173655*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_173655AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/173655*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¦
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????A
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????A
8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????A
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
υ%

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_173014

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_172999
assignmovingavg_1_173006
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
I:+???????????????????????????8:8:8:8:8:*
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
loc:@AssignMovingAvg/172999*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/172999*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_172999*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/172999*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/172999*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_172999AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/172999*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/173006*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173006*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_173006*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173006*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173006*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_173006AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/173006*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpΈ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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

§
4__inference_batch_normalization_layer_call_fn_175268

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1735902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????8::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????8
 
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
Ι
§
4__inference_batch_normalization_layer_call_fn_175173

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallι
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????8**
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1726782
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
ι
L
0__inference_max_pooling2d_2_layer_call_fn_173072

inputs
identity¨
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1730662
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ε
©
6__inference_batch_normalization_5_layer_call_fn_176045

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΡ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1735172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
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
ι
L
0__inference_max_pooling2d_1_layer_call_fn_172904

inputs
identity¨
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1728982
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
σ%

O__inference_batch_normalization_layer_call_and_return_conditional_losses_172678

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_172663
assignmovingavg_1_172670
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
I:+???????????????????????????8:8:8:8:8:*
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
loc:@AssignMovingAvg/172663*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/172663*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_172663*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/172663*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/172663*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_172663AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/172663*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/172670*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172670*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_172670*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172670*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172670*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_172670AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/172670*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpΈ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_172898

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
­%

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_175314

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_175299
assignmovingavg_1_175306
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
7:?????????A
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
loc:@AssignMovingAvg/175299*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/175299*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_175299*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/175299*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/175299*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_175299AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/175299*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/175306*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175306*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_175306*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175306*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175306*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_175306AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/175306*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¦
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????A
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????A
8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????A
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
Ί
ς
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172713

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constμ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
φ

F__inference_sequential_layer_call_and_return_conditional_losses_174823

inputs/
+quant_conv2d_conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource.
*batch_normalization_assignmovingavg_1746220
,batch_normalization_assignmovingavg_1_1746291
-quant_conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resource0
,batch_normalization_1_assignmovingavg_1746562
.batch_normalization_1_assignmovingavg_1_1746631
-quant_conv2d_2_conv2d_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resource0
,batch_normalization_2_assignmovingavg_1746902
.batch_normalization_2_assignmovingavg_1_174697.
*quant_dense_matmul_readvariableop_resource0
,batch_normalization_3_assignmovingavg_1747212
.batch_normalization_3_assignmovingavg_1_174727?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource0
,quant_dense_1_matmul_readvariableop_resource0
,batch_normalization_4_assignmovingavg_1747592
.batch_normalization_4_assignmovingavg_1_174765?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource0
,quant_dense_2_matmul_readvariableop_resource0
,batch_normalization_5_assignmovingavg_1747972
.batch_normalization_5_assignmovingavg_1_174803?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource
identity’7batch_normalization/AssignMovingAvg/AssignSubVariableOp’2batch_normalization/AssignMovingAvg/ReadVariableOp’9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp’4batch_normalization/AssignMovingAvg_1/ReadVariableOp’"batch_normalization/ReadVariableOp’$batch_normalization/ReadVariableOp_1’9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp’4batch_normalization_1/AssignMovingAvg/ReadVariableOp’;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp’6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp’$batch_normalization_1/ReadVariableOp’&batch_normalization_1/ReadVariableOp_1’9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp’4batch_normalization_2/AssignMovingAvg/ReadVariableOp’;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp’6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp’$batch_normalization_2/ReadVariableOp’&batch_normalization_2/ReadVariableOp_1’9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp’4batch_normalization_3/AssignMovingAvg/ReadVariableOp’;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp’6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp’.batch_normalization_3/batchnorm/ReadVariableOp’2batch_normalization_3/batchnorm/mul/ReadVariableOp’9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp’4batch_normalization_4/AssignMovingAvg/ReadVariableOp’;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp’6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp’.batch_normalization_4/batchnorm/ReadVariableOp’2batch_normalization_4/batchnorm/mul/ReadVariableOp’9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp’4batch_normalization_5/AssignMovingAvg/ReadVariableOp’;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp’6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp’.batch_normalization_5/batchnorm/ReadVariableOp’2batch_normalization_5/batchnorm/mul/ReadVariableOp’"quant_conv2d/Conv2D/ReadVariableOp’$quant_conv2d_1/Conv2D/ReadVariableOp’$quant_conv2d_2/Conv2D/ReadVariableOp’!quant_dense/MatMul/ReadVariableOp’#quant_dense_1/MatMul/ReadVariableOp’#quant_dense_2/MatMul/ReadVariableOpΌ
"quant_conv2d/Conv2D/ReadVariableOpReadVariableOp+quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02$
"quant_conv2d/Conv2D/ReadVariableOpΛ
quant_conv2d/Conv2DConv2Dinputs*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????8*
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
 batch_normalization/LogicalAnd/yΌ
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2 
batch_normalization/LogicalAnd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype02$
"batch_normalization/ReadVariableOpΆ
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
8:?????????8:8:8:8:8:*
epsilon%o:2&
$batch_normalization/FusedBatchNormV3
batch_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization/Const_2Ϊ
)batch_normalization/AssignMovingAvg/sub/xConst*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/174622*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)batch_normalization/AssignMovingAvg/sub/x
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/174622*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/subΟ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_assignmovingavg_174622*
_output_shapes
:8*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp°
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/174622*
_output_shapes
:82+
)batch_normalization/AssignMovingAvg/sub_1
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/174622*
_output_shapes
:82)
'batch_normalization/AssignMovingAvg/mulω
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_assignmovingavg_174622+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/174622*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpΰ
+batch_normalization/AssignMovingAvg_1/sub/xConst*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/174629*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization/AssignMovingAvg_1/sub/x
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/174629*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/subΥ
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_assignmovingavg_1_174629*
_output_shapes
:8*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpΌ
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/174629*
_output_shapes
:82-
+batch_normalization/AssignMovingAvg_1/sub_1£
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/174629*
_output_shapes
:82+
)batch_normalization/AssignMovingAvg_1/mul
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_assignmovingavg_1_174629-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/174629*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpΠ
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????A
8*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolΒ
$quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_1/Conv2D/ReadVariableOpθ
quant_conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????A
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
"batch_normalization_1/LogicalAnd/yΔ
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_1/LogicalAndΆ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:8*
dtype02&
$batch_normalization_1/ReadVariableOpΌ
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
batch_normalization_1/Const_1‘
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3quant_conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0$batch_normalization_1/Const:output:0&batch_normalization_1/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:?????????A
8:8:8:8:8:*
epsilon%o:2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_1/Const_2ΰ
+batch_normalization_1/AssignMovingAvg/sub/xConst*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/174656*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_1/AssignMovingAvg/sub/x
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/174656*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/subΥ
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_1_assignmovingavg_174656*
_output_shapes
:8*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpΊ
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/174656*
_output_shapes
:82-
+batch_normalization_1/AssignMovingAvg/sub_1£
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/174656*
_output_shapes
:82+
)batch_normalization_1/AssignMovingAvg/mul
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_1_assignmovingavg_174656-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/174656*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpζ
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/174663*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_1/AssignMovingAvg_1/sub/x₯
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/174663*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/subΫ
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_1_assignmovingavg_1_174663*
_output_shapes
:8*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpΖ
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/174663*
_output_shapes
:82/
-batch_normalization_1/AssignMovingAvg_1/sub_1­
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/174663*
_output_shapes
:82-
+batch_normalization_1/AssignMovingAvg_1/mul
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_1_assignmovingavg_1_174663/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/174663*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpΦ
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:????????? 8*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolΒ
$quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_2/Conv2D/ReadVariableOpκ
quant_conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 8*
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
"batch_normalization_2/LogicalAnd/yΔ
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAndΆ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:8*
dtype02&
$batch_normalization_2/ReadVariableOpΌ
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
batch_normalization_2/Const_1‘
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3quant_conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0$batch_normalization_2/Const:output:0&batch_normalization_2/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:????????? 8:8:8:8:8:*
epsilon%o:2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_2/Const_2ΰ
+batch_normalization_2/AssignMovingAvg/sub/xConst*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/174690*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+batch_normalization_2/AssignMovingAvg/sub/x
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/174690*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/subΥ
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_2_assignmovingavg_174690*
_output_shapes
:8*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpΊ
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/174690*
_output_shapes
:82-
+batch_normalization_2/AssignMovingAvg/sub_1£
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/174690*
_output_shapes
:82+
)batch_normalization_2/AssignMovingAvg/mul
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_2_assignmovingavg_174690-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/174690*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpζ
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/174697*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-batch_normalization_2/AssignMovingAvg_1/sub/x₯
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/174697*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/subΫ
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_2_assignmovingavg_1_174697*
_output_shapes
:8*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpΖ
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/174697*
_output_shapes
:82/
-batch_normalization_2/AssignMovingAvg_1/sub_1­
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/174697*
_output_shapes
:82-
+batch_normalization_2/AssignMovingAvg_1/mul
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_2_assignmovingavg_1_174697/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/174697*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpΦ
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????8*
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
valueB"????   2
flatten/Const
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
flatten/Reshape³
!quant_dense/MatMul/ReadVariableOpReadVariableOp*quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!quant_dense/MatMul/ReadVariableOpͺ
quant_dense/MatMulMatMulflatten/Reshape:output:0)quant_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
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
"batch_normalization_3/LogicalAnd/yΔ
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAndΆ
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesθ
"batch_normalization_3/moments/meanMeanquant_dense/MatMul:product:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2$
"batch_normalization_3/moments/meanΏ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:	2,
*batch_normalization_3/moments/StopGradientύ
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencequant_dense/MatMul:product:03batch_normalization_3/moments/StopGradient:output:0*
T0*(
_output_shapes
:?????????21
/batch_normalization_3/moments/SquaredDifferenceΎ
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
&batch_normalization_3/moments/varianceΓ
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2'
%batch_normalization_3/moments/SqueezeΛ
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1ΰ
+batch_normalization_3/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/174721*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2-
+batch_normalization_3/AssignMovingAvg/decayΦ
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_3_assignmovingavg_174721*
_output_shapes	
:*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp²
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/174721*
_output_shapes	
:2+
)batch_normalization_3/AssignMovingAvg/sub©
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/174721*
_output_shapes	
:2+
)batch_normalization_3/AssignMovingAvg/mul
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_3_assignmovingavg_174721-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/174721*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpζ
-batch_normalization_3/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/174727*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2/
-batch_normalization_3/AssignMovingAvg_1/decayά
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_3_assignmovingavg_1_174727*
_output_shapes	
:*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpΌ
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/174727*
_output_shapes	
:2-
+batch_normalization_3/AssignMovingAvg_1/sub³
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/174727*
_output_shapes	
:2-
+batch_normalization_3/AssignMovingAvg_1/mul
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_3_assignmovingavg_1_174727/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/174727*
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
%batch_normalization_3/batchnorm/add/yΫ
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/add¦
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_3/batchnorm/Rsqrtα
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpή
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/mulΟ
%batch_normalization_3/batchnorm/mul_1Mulquant_dense/MatMul:product:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:?????????2'
%batch_normalization_3/batchnorm/mul_1Τ
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_3/batchnorm/mul_2Υ
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpΪ
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_3/batchnorm/subή
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????2'
%batch_normalization_3/batchnorm/add_1Ή
#quant_dense_1/MatMul/ReadVariableOpReadVariableOp,quant_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#quant_dense_1/MatMul/ReadVariableOpΑ
quant_dense_1/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0+quant_dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
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
"batch_normalization_4/LogicalAnd/yΔ
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_4/LogicalAndΆ
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesκ
"batch_normalization_4/moments/meanMeanquant_dense_1/MatMul:product:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2$
"batch_normalization_4/moments/meanΏ
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	2,
*batch_normalization_4/moments/StopGradient?
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencequant_dense_1/MatMul:product:03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:?????????21
/batch_normalization_4/moments/SquaredDifferenceΎ
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
&batch_normalization_4/moments/varianceΓ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2'
%batch_normalization_4/moments/SqueezeΛ
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1ΰ
+batch_normalization_4/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/174759*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2-
+batch_normalization_4/AssignMovingAvg/decayΦ
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_4_assignmovingavg_174759*
_output_shapes	
:*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp²
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/174759*
_output_shapes	
:2+
)batch_normalization_4/AssignMovingAvg/sub©
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/174759*
_output_shapes	
:2+
)batch_normalization_4/AssignMovingAvg/mul
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_4_assignmovingavg_174759-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/174759*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpζ
-batch_normalization_4/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/174765*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2/
-batch_normalization_4/AssignMovingAvg_1/decayά
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_4_assignmovingavg_1_174765*
_output_shapes	
:*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpΌ
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/174765*
_output_shapes	
:2-
+batch_normalization_4/AssignMovingAvg_1/sub³
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/174765*
_output_shapes	
:2-
+batch_normalization_4/AssignMovingAvg_1/mul
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_4_assignmovingavg_1_174765/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/174765*
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
%batch_normalization_4/batchnorm/add/yΫ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/add¦
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_4/batchnorm/Rsqrtα
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpή
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/mulΡ
%batch_normalization_4/batchnorm/mul_1Mulquant_dense_1/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:?????????2'
%batch_normalization_4/batchnorm/mul_1Τ
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_4/batchnorm/mul_2Υ
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpΪ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2%
#batch_normalization_4/batchnorm/subή
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:?????????2'
%batch_normalization_4/batchnorm/add_1Έ
#quant_dense_2/MatMul/ReadVariableOpReadVariableOp,quant_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#quant_dense_2/MatMul/ReadVariableOpΐ
quant_dense_2/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0+quant_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
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
"batch_normalization_5/LogicalAnd/yΔ
 batch_normalization_5/LogicalAnd
LogicalAnd+batch_normalization_5/LogicalAnd/x:output:0+batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_5/LogicalAndΆ
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_5/moments/mean/reduction_indicesι
"batch_normalization_5/moments/meanMeanquant_dense_2/MatMul:product:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_5/moments/meanΎ
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_5/moments/StopGradientώ
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencequant_dense_2/MatMul:product:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????21
/batch_normalization_5/moments/SquaredDifferenceΎ
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
&batch_normalization_5/moments/varianceΒ
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_5/moments/SqueezeΚ
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1ΰ
+batch_normalization_5/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/174797*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2-
+batch_normalization_5/AssignMovingAvg/decayΥ
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_5_assignmovingavg_174797*
_output_shapes
:*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOp±
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/174797*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/sub¨
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/174797*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/mul
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_5_assignmovingavg_174797-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/174797*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpζ
-batch_normalization_5/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/174803*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2/
-batch_normalization_5/AssignMovingAvg_1/decayΫ
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_5_assignmovingavg_1_174803*
_output_shapes
:*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp»
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/174803*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/sub²
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/174803*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/mul
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_5_assignmovingavg_1_174803/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/174803*
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
%batch_normalization_5/batchnorm/add/yΪ
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/add₯
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtΰ
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpέ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mulΠ
%batch_normalization_5/batchnorm/mul_1Mulquant_dense_2/MatMul:product:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_5/batchnorm/mul_1Σ
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2Τ
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpΩ
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/subέ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2'
%batch_normalization_5/batchnorm/add_1
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
activation/Softmaxε
IdentityIdentityactivation/Softmax:softmax:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp#^quant_conv2d/Conv2D/ReadVariableOp%^quant_conv2d_1/Conv2D/ReadVariableOp%^quant_conv2d_2/Conv2D/ReadVariableOp"^quant_dense/MatMul/ReadVariableOp$^quant_dense_1/MatMul/ReadVariableOp$^quant_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::2r
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
:?????????
 
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
σ%

O__inference_batch_normalization_layer_call_and_return_conditional_losses_175138

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_175123
assignmovingavg_1_175130
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
I:+???????????????????????????8:8:8:8:8:*
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
loc:@AssignMovingAvg/175123*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/175123*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_175123*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/175123*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/175123*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_175123AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/175123*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/175130*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175130*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_175130*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175130*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175130*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_175130AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/175130*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpΈ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
τ
τ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_175336

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????A
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
ConstΪ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????A
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????A
8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????A
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
Y
΅
F__inference_sequential_layer_call_and_return_conditional_losses_174172

inputs
quant_conv2d_174094
batch_normalization_174097
batch_normalization_174099
batch_normalization_174101
batch_normalization_174103
quant_conv2d_1_174107 
batch_normalization_1_174110 
batch_normalization_1_174112 
batch_normalization_1_174114 
batch_normalization_1_174116
quant_conv2d_2_174120 
batch_normalization_2_174123 
batch_normalization_2_174125 
batch_normalization_2_174127 
batch_normalization_2_174129
quant_dense_174134 
batch_normalization_3_174137 
batch_normalization_3_174139 
batch_normalization_3_174141 
batch_normalization_3_174143
quant_dense_1_174146 
batch_normalization_4_174149 
batch_normalization_4_174151 
batch_normalization_4_174153 
batch_normalization_4_174155
quant_dense_2_174158 
batch_normalization_5_174161 
batch_normalization_5_174163 
batch_normalization_5_174165 
batch_normalization_5_174167
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’-batch_normalization_4/StatefulPartitionedCall’-batch_normalization_5/StatefulPartitionedCall’$quant_conv2d/StatefulPartitionedCall’&quant_conv2d_1/StatefulPartitionedCall’&quant_conv2d_2/StatefulPartitionedCall’#quant_dense/StatefulPartitionedCall’%quant_dense_1/StatefulPartitionedCall’%quant_dense_2/StatefulPartitionedCallΣ
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsquant_conv2d_174094*
Tin
2*
Tout
2*0
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1725762&
$quant_conv2d/StatefulPartitionedCallν
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_174097batch_normalization_174099batch_normalization_174101batch_normalization_174103*
Tin	
2*
Tout
2*0
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1735682-
+batch_normalization/StatefulPartitionedCallΥ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1727302
max_pooling2d/PartitionedCallϊ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_174107*
Tin
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1727442(
&quant_conv2d_1/StatefulPartitionedCallό
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_174110batch_normalization_1_174112batch_normalization_1_174114batch_normalization_1_174116*
Tin	
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1736632/
-batch_normalization_1/StatefulPartitionedCallέ
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1728982!
max_pooling2d_1/PartitionedCallό
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_174120*
Tin
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1729122(
&quant_conv2d_2/StatefulPartitionedCallό
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_174123batch_normalization_2_174125batch_normalization_2_174127batch_normalization_2_174129*
Tin	
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1737582/
-batch_normalization_2/StatefulPartitionedCallέ
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1730662!
max_pooling2d_2/PartitionedCall°
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1738232
flatten/PartitionedCallα
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_174134*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_1738382%
#quant_dense/StatefulPartitionedCallς
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_174137batch_normalization_3_174139batch_normalization_3_174141batch_normalization_3_174143*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1731772/
-batch_normalization_3/StatefulPartitionedCall?
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_174146*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1738922'
%quant_dense_1/StatefulPartitionedCallτ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_174149batch_normalization_4_174151batch_normalization_4_174153batch_normalization_4_174155*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1733292/
-batch_normalization_4/StatefulPartitionedCallώ
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_174158*
Tin
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1739462'
%quant_dense_2/StatefulPartitionedCallσ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_174161batch_normalization_5_174163batch_normalization_5_174165batch_normalization_5_174167*
Tin	
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1734812/
-batch_normalization_5/StatefulPartitionedCallΖ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1739982
activation/PartitionedCall
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::2Z
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
:?????????
 
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
ι
©
6__inference_batch_normalization_3_layer_call_fn_175769

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1732132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
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
?
G
+__inference_activation_layer_call_fn_176055

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1739982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
φ
ς
O__inference_batch_normalization_layer_call_and_return_conditional_losses_175242

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1Λ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstΫ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????8
 
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
?
₯
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_172744

inputs"
conv2d_readvariableop_resource
identity’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????8*
paddingSAME*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????8:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????8
 
_user_specified_nameinputs:

_output_shapes
: 


Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_173213

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp^
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
:?????????2
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
:?????????2
batchnorm/add_1ά
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
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

t
.__inference_quant_dense_2_layer_call_fn_175921

inputs
unknown
identity’StatefulPartitionedCall₯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1739462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
φ
ς
O__inference_batch_normalization_layer_call_and_return_conditional_losses_173590

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1Λ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstΫ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????8
 
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
ϊ

Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_173517

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp^
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
:?????????2
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
:?????????2
batchnorm/add_1Ϋ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
Υ0
Θ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_175858

inputs
assignmovingavg_175833
assignmovingavg_1_175839)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp^
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
moments/StopGradient₯
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:?????????2
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
loc:@AssignMovingAvg/175833*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_175833*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpΔ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/175833*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/175833*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_175833AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/175833*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/175839*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_175839*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΞ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175839*
_output_shapes	
:2
AssignMovingAvg_1/subΕ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175839*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_175839AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/175839*
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
:?????????2
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
:?????????2
batchnorm/add_1΄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
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
Ό
b
F__inference_activation_layer_call_and_return_conditional_losses_173998

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_173066

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ό
·
+__inference_sequential_layer_call_fn_175027

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
identity’StatefulPartitionedCall
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
:?????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1741722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
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
½0
Θ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_173481

inputs
assignmovingavg_173456
assignmovingavg_1_173462)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp^
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
moments/StopGradient€
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
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
loc:@AssignMovingAvg/173456*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_173456*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpΓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/173456*
_output_shapes
:2
AssignMovingAvg/subΊ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/173456*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_173456AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/173456*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/173462*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_173462*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΝ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173462*
_output_shapes
:2
AssignMovingAvg_1/subΔ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173462*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_173462AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/173462*
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
:?????????2
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
:?????????2
batchnorm/add_1³
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
―%

O__inference_batch_normalization_layer_call_and_return_conditional_losses_173568

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_173553
assignmovingavg_1_173560
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
8:?????????8:8:8:8:8:*
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
loc:@AssignMovingAvg/173553*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/173553*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_173553*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/173553*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/173553*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_173553AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/173553*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/173560*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173560*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_173560*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173560*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173560*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_173560AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/173560*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp§
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????8
 
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

u
/__inference_quant_conv2d_2_layer_call_fn_172920

inputs
unknown
identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????8**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1729122
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????8:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????8
 
_user_specified_nameinputs:

_output_shapes
: 

s
-__inference_quant_conv2d_layer_call_fn_172584

inputs
unknown
identity’StatefulPartitionedCallΎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????8**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1725762
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
₯
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_172912

inputs"
conv2d_readvariableop_resource
identity’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????8*
paddingSAME*
strides
2
Conv2D
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????8:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????8
 
_user_specified_nameinputs:

_output_shapes
: 
ε
J
.__inference_max_pooling2d_layer_call_fn_172736

inputs
identity¦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1727302
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

’
G__inference_quant_dense_layer_call_and_return_conditional_losses_175638

inputs"
matmul_readvariableop_resource
identity’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
Ν
©
6__inference_batch_normalization_1_layer_call_fn_175444

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallλ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1728812
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
Υ0
Θ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_173329

inputs
assignmovingavg_173304
assignmovingavg_1_173310)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp^
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
moments/StopGradient₯
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:?????????2
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
loc:@AssignMovingAvg/173304*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_173304*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpΔ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/173304*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/173304*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_173304AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/173304*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/173310*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_173310*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΞ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173310*
_output_shapes	
:2
AssignMovingAvg_1/subΕ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173310*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_173310AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/173310*
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
:?????????2
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
:?????????2
batchnorm/add_1΄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
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

©
6__inference_batch_normalization_1_layer_call_fn_175362

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1736852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????A
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????A
8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????A
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

§
4__inference_batch_normalization_layer_call_fn_175255

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1735682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????8::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????8
 
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
C__inference_flatten_layer_call_and_return_conditional_losses_173823

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????8:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
ε
©
6__inference_batch_normalization_5_layer_call_fn_176032

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΡ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1734812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
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

r
,__inference_quant_dense_layer_call_fn_175645

inputs
unknown
identity’StatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_1738382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
«ϊ
?
!__inference__wrapped_model_172568
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
identity’>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp’@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1’-sequential/batch_normalization/ReadVariableOp’/sequential/batch_normalization/ReadVariableOp_1’@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp’Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1’/sequential/batch_normalization_1/ReadVariableOp’1sequential/batch_normalization_1/ReadVariableOp_1’@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp’Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1’/sequential/batch_normalization_2/ReadVariableOp’1sequential/batch_normalization_2/ReadVariableOp_1’9sequential/batch_normalization_3/batchnorm/ReadVariableOp’;sequential/batch_normalization_3/batchnorm/ReadVariableOp_1’;sequential/batch_normalization_3/batchnorm/ReadVariableOp_2’=sequential/batch_normalization_3/batchnorm/mul/ReadVariableOp’9sequential/batch_normalization_4/batchnorm/ReadVariableOp’;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1’;sequential/batch_normalization_4/batchnorm/ReadVariableOp_2’=sequential/batch_normalization_4/batchnorm/mul/ReadVariableOp’9sequential/batch_normalization_5/batchnorm/ReadVariableOp’;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1’;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2’=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOp’-sequential/quant_conv2d/Conv2D/ReadVariableOp’/sequential/quant_conv2d_1/Conv2D/ReadVariableOp’/sequential/quant_conv2d_2/Conv2D/ReadVariableOp’,sequential/quant_dense/MatMul/ReadVariableOp’.sequential/quant_dense_1/MatMul/ReadVariableOp’.sequential/quant_dense_2/MatMul/ReadVariableOpέ
-sequential/quant_conv2d/Conv2D/ReadVariableOpReadVariableOp6sequential_quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02/
-sequential/quant_conv2d/Conv2D/ReadVariableOpψ
sequential/quant_conv2d/Conv2DConv2Dquant_conv2d_input5sequential/quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????8*
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
+sequential/batch_normalization/LogicalAnd/yθ
)sequential/batch_normalization/LogicalAnd
LogicalAnd4sequential/batch_normalization/LogicalAnd/x:output:04sequential/batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2+
)sequential/batch_normalization/LogicalAndΡ
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:8*
dtype02/
-sequential/batch_normalization/ReadVariableOpΧ
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
8:?????????8:8:8:8:8:*
epsilon%o:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3
$sequential/batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2&
$sequential/batch_normalization/Constρ
 sequential/max_pooling2d/MaxPoolMaxPool3sequential/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:?????????A
8*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolγ
/sequential/quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp8sequential_quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype021
/sequential/quant_conv2d_1/Conv2D/ReadVariableOp
 sequential/quant_conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:07sequential/quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????A
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
-sequential/batch_normalization_1/LogicalAnd/yπ
+sequential/batch_normalization_1/LogicalAnd
LogicalAnd6sequential/batch_normalization_1/LogicalAnd/x:output:06sequential/batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_1/LogicalAndΧ
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization_1/ReadVariableOpέ
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
7:?????????A
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
&sequential/batch_normalization_1/Constχ
"sequential/max_pooling2d_1/MaxPoolMaxPool5sequential/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:????????? 8*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPoolγ
/sequential/quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp8sequential_quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype021
/sequential/quant_conv2d_2/Conv2D/ReadVariableOp
 sequential/quant_conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:07sequential/quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 8*
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
-sequential/batch_normalization_2/LogicalAnd/yπ
+sequential/batch_normalization_2/LogicalAnd
LogicalAnd6sequential/batch_normalization_2/LogicalAnd/x:output:06sequential/batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_2/LogicalAndΧ
/sequential/batch_normalization_2/ReadVariableOpReadVariableOp8sequential_batch_normalization_2_readvariableop_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization_2/ReadVariableOpέ
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
7:????????? 8:8:8:8:8:*
epsilon%o:*
is_training( 23
1sequential/batch_normalization_2/FusedBatchNormV3
&sequential/batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2(
&sequential/batch_normalization_2/Constχ
"sequential/max_pooling2d_2/MaxPoolMaxPool5sequential/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:?????????8*
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
valueB"????   2
sequential/flatten/ConstΖ
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_2/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
sequential/flatten/ReshapeΤ
,sequential/quant_dense/MatMul/ReadVariableOpReadVariableOp5sequential_quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,sequential/quant_dense/MatMul/ReadVariableOpΦ
sequential/quant_dense/MatMulMatMul#sequential/flatten/Reshape:output:04sequential/quant_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
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
-sequential/batch_normalization_3/LogicalAnd/yπ
+sequential/batch_normalization_3/LogicalAnd
LogicalAnd6sequential/batch_normalization_3/LogicalAnd/x:output:06sequential/batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_3/LogicalAndφ
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
.sequential/batch_normalization_3/batchnorm/addΗ
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
.sequential/batch_normalization_3/batchnorm/mulϋ
0sequential/batch_normalization_3/batchnorm/mul_1Mul'sequential/quant_dense/MatMul:product:02sequential/batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:?????????22
0sequential/batch_normalization_3/batchnorm/mul_1ό
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02=
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_1
0sequential/batch_normalization_3/batchnorm/mul_2MulCsequential/batch_normalization_3/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:22
0sequential/batch_normalization_3/batchnorm/mul_2ό
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
:?????????22
0sequential/batch_normalization_3/batchnorm/add_1Ϊ
.sequential/quant_dense_1/MatMul/ReadVariableOpReadVariableOp7sequential_quant_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype020
.sequential/quant_dense_1/MatMul/ReadVariableOpν
sequential/quant_dense_1/MatMulMatMul4sequential/batch_normalization_3/batchnorm/add_1:z:06sequential/quant_dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2!
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
-sequential/batch_normalization_4/LogicalAnd/yπ
+sequential/batch_normalization_4/LogicalAnd
LogicalAnd6sequential/batch_normalization_4/LogicalAnd/x:output:06sequential/batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_4/LogicalAndφ
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
.sequential/batch_normalization_4/batchnorm/addΗ
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
.sequential/batch_normalization_4/batchnorm/mulύ
0sequential/batch_normalization_4/batchnorm/mul_1Mul)sequential/quant_dense_1/MatMul:product:02sequential/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:?????????22
0sequential/batch_normalization_4/batchnorm/mul_1ό
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02=
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1
0sequential/batch_normalization_4/batchnorm/mul_2MulCsequential/batch_normalization_4/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:22
0sequential/batch_normalization_4/batchnorm/mul_2ό
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
:?????????22
0sequential/batch_normalization_4/batchnorm/add_1Ω
.sequential/quant_dense_2/MatMul/ReadVariableOpReadVariableOp7sequential_quant_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype020
.sequential/quant_dense_2/MatMul/ReadVariableOpμ
sequential/quant_dense_2/MatMulMatMul4sequential/batch_normalization_4/batchnorm/add_1:z:06sequential/quant_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
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
-sequential/batch_normalization_5/LogicalAnd/yπ
+sequential/batch_normalization_5/LogicalAnd
LogicalAnd6sequential/batch_normalization_5/LogicalAnd/x:output:06sequential/batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_5/LogicalAndυ
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
.sequential/batch_normalization_5/batchnorm/addΖ
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
.sequential/batch_normalization_5/batchnorm/mulό
0sequential/batch_normalization_5/batchnorm/mul_1Mul)sequential/quant_dense_2/MatMul:product:02sequential/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????22
0sequential/batch_normalization_5/batchnorm/mul_1ϋ
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1
0sequential/batch_normalization_5/batchnorm/mul_2MulCsequential/batch_normalization_5/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:22
0sequential/batch_normalization_5/batchnorm/mul_2ϋ
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
:?????????22
0sequential/batch_normalization_5/batchnorm/add_1±
sequential/activation/SoftmaxSoftmax4sequential/batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2
sequential/activation/SoftmaxΚ
IdentityIdentity'sequential/activation/Softmax:softmax:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1A^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_2/ReadVariableOp2^sequential/batch_normalization_2/ReadVariableOp_1:^sequential/batch_normalization_3/batchnorm/ReadVariableOp<^sequential/batch_normalization_3/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_3/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_3/batchnorm/mul/ReadVariableOp:^sequential/batch_normalization_4/batchnorm/ReadVariableOp<^sequential/batch_normalization_4/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_4/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_4/batchnorm/mul/ReadVariableOp:^sequential/batch_normalization_5/batchnorm/ReadVariableOp<^sequential/batch_normalization_5/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_5/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_5/batchnorm/mul/ReadVariableOp.^sequential/quant_conv2d/Conv2D/ReadVariableOp0^sequential/quant_conv2d_1/Conv2D/ReadVariableOp0^sequential/quant_conv2d_2/Conv2D/ReadVariableOp-^sequential/quant_dense/MatMul/ReadVariableOp/^sequential/quant_dense_1/MatMul/ReadVariableOp/^sequential/quant_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::2
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
:?????????
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
Ό
τ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172881

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constμ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
ΓY
Α
F__inference_sequential_layer_call_and_return_conditional_losses_174007
quant_conv2d_input
quant_conv2d_173532
batch_normalization_173617
batch_normalization_173619
batch_normalization_173621
batch_normalization_173623
quant_conv2d_1_173627 
batch_normalization_1_173712 
batch_normalization_1_173714 
batch_normalization_1_173716 
batch_normalization_1_173718
quant_conv2d_2_173722 
batch_normalization_2_173807 
batch_normalization_2_173809 
batch_normalization_2_173811 
batch_normalization_2_173813
quant_dense_173847 
batch_normalization_3_173876 
batch_normalization_3_173878 
batch_normalization_3_173880 
batch_normalization_3_173882
quant_dense_1_173901 
batch_normalization_4_173930 
batch_normalization_4_173932 
batch_normalization_4_173934 
batch_normalization_4_173936
quant_dense_2_173955 
batch_normalization_5_173984 
batch_normalization_5_173986 
batch_normalization_5_173988 
batch_normalization_5_173990
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’-batch_normalization_4/StatefulPartitionedCall’-batch_normalization_5/StatefulPartitionedCall’$quant_conv2d/StatefulPartitionedCall’&quant_conv2d_1/StatefulPartitionedCall’&quant_conv2d_2/StatefulPartitionedCall’#quant_dense/StatefulPartitionedCall’%quant_dense_1/StatefulPartitionedCall’%quant_dense_2/StatefulPartitionedCallί
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputquant_conv2d_173532*
Tin
2*
Tout
2*0
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1725762&
$quant_conv2d/StatefulPartitionedCallν
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_173617batch_normalization_173619batch_normalization_173621batch_normalization_173623*
Tin	
2*
Tout
2*0
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1735682-
+batch_normalization/StatefulPartitionedCallΥ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1727302
max_pooling2d/PartitionedCallϊ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_173627*
Tin
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1727442(
&quant_conv2d_1/StatefulPartitionedCallό
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_173712batch_normalization_1_173714batch_normalization_1_173716batch_normalization_1_173718*
Tin	
2*
Tout
2*/
_output_shapes
:?????????A
8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1736632/
-batch_normalization_1/StatefulPartitionedCallέ
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1728982!
max_pooling2d_1/PartitionedCallό
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_173722*
Tin
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1729122(
&quant_conv2d_2/StatefulPartitionedCallό
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_173807batch_normalization_2_173809batch_normalization_2_173811batch_normalization_2_173813*
Tin	
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1737582/
-batch_normalization_2/StatefulPartitionedCallέ
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????8**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1730662!
max_pooling2d_2/PartitionedCall°
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1738232
flatten/PartitionedCallα
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_173847*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_1738382%
#quant_dense/StatefulPartitionedCallς
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_173876batch_normalization_3_173878batch_normalization_3_173880batch_normalization_3_173882*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1731772/
-batch_normalization_3/StatefulPartitionedCall?
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_173901*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1738922'
%quant_dense_1/StatefulPartitionedCallτ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_173930batch_normalization_4_173932batch_normalization_4_173934batch_normalization_4_173936*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1733292/
-batch_normalization_4/StatefulPartitionedCallώ
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_173955*
Tin
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1739462'
%quant_dense_2/StatefulPartitionedCallσ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_173984batch_normalization_5_173986batch_normalization_5_173988batch_normalization_5_173990*
Tin	
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1734812/
-batch_normalization_5/StatefulPartitionedCallΖ
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1739982
activation/PartitionedCall
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::2Z
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
:?????????
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
Ό
τ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_175418

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constμ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
―%

O__inference_batch_normalization_layer_call_and_return_conditional_losses_175220

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_175205
assignmovingavg_1_175212
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
8:?????????8:8:8:8:8:*
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
loc:@AssignMovingAvg/175205*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/175205*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_175205*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/175205*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/175205*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_175205AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/175205*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/175212*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175212*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_175212*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175212*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175212*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_175212AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/175212*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp§
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????8
 
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

’
G__inference_quant_dense_layer_call_and_return_conditional_losses_173838

inputs"
matmul_readvariableop_resource
identity’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
τ
τ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_173780

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? 8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstΪ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? 8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 8
 
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
υ%

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175490

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_175475
assignmovingavg_1_175482
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
I:+???????????????????????????8:8:8:8:8:*
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
loc:@AssignMovingAvg/175475*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/175475*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_175475*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/175475*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/175475*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_175475AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/175475*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/175482*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175482*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_175482*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175482*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175482*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_175482AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/175482*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpΈ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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

©
6__inference_batch_normalization_2_layer_call_fn_175607

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:????????? 8**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1737582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? 8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 8
 
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
υ%

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_175396

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_175381
assignmovingavg_1_175388
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
I:+???????????????????????????8:8:8:8:8:*
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
loc:@AssignMovingAvg/175381*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/175381*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_175381*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/175381*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/175381*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_175381AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/175381*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/175388*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175388*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_175388*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175388*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175388*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_175388AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/175388*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpΈ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
Υ0
Θ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_175720

inputs
assignmovingavg_175695
assignmovingavg_1_175701)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOp^
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
moments/StopGradient₯
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:?????????2
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
loc:@AssignMovingAvg/175695*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_175695*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpΔ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/175695*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/175695*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_175695AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/175695*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/175701*
_output_shapes
: *
dtype0*
valueB
 *ΝΜΜ=2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_175701*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΞ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175701*
_output_shapes	
:2
AssignMovingAvg_1/subΕ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/175701*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_175701AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/175701*
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
:?????????2
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
:?????????2
batchnorm/add_1΄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
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
υ%

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172846

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_172831
assignmovingavg_1_172838
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
I:+???????????????????????????8:8:8:8:8:*
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
loc:@AssignMovingAvg/172831*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/172831*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_172831*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/172831*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/172831*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_172831AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/172831*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/172838*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172838*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_172838*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172838*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172838*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_172838AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/172838*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpΈ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
Ό
τ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175512

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????8:8:8:8:8:*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constμ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????8
 
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_173758

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_173743
assignmovingavg_1_173750
identity’#AssignMovingAvg/AssignSubVariableOp’AssignMovingAvg/ReadVariableOp’%AssignMovingAvg_1/AssignSubVariableOp’ AssignMovingAvg_1/ReadVariableOp’ReadVariableOp’ReadVariableOp_1^
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
7:????????? 8:8:8:8:8:*
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
loc:@AssignMovingAvg/173743*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/x―
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/173743*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_173743*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpΜ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/173743*
_output_shapes
:82
AssignMovingAvg/sub_1΅
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/173743*
_output_shapes
:82
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_173743AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/173743*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp€
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/173750*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/x·
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173750*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_173750*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpΨ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173750*
_output_shapes
:82
AssignMovingAvg_1/sub_1Ώ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173750*
_output_shapes
:82
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_173750AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/173750*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¦
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? 8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 8
 
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
τ
τ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_173685

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1^
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
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????A
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
ConstΪ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????A
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????A
8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????A
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_175743

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp^
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
:?????????2
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
:?????????2
batchnorm/add_1ά
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
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

€
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_173946

inputs"
matmul_readvariableop_resource
identity’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
ι
©
6__inference_batch_normalization_3_layer_call_fn_175756

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1731772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
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
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_172730

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
‘
t
.__inference_quant_dense_1_layer_call_fn_175783

inputs
unknown
identity’StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1738922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
Ό
b
F__inference_activation_layer_call_and_return_conditional_losses_176050

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs


Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_173365

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOp^
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
:?????????2
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
:?????????2
batchnorm/add_1ά
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
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
ΰ
Γ
+__inference_sequential_layer_call_fn_174381
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
identity’StatefulPartitionedCall
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
:?????????**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1743182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*©
_input_shapes
:?????????::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:?????????
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
: "―L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Μ
serving_defaultΈ
Z
quant_conv2d_inputD
$serving_default_quant_conv2d_input:0?????????>

activation0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ΆΘ
Υ
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"ι
_tf_keras_sequentialΚ{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "QuantConv2D", "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "QuantDense", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}], "build_input_shape": [null, 130, 20, 1]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "QuantConv2D", "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "QuantDense", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "input_quantizer": null, "kernel_quantizer": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}], "build_input_shape": [null, 130, 20, 1]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": ["accuracy", "sparse_categorical_accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0013894954463467002, "decay": 0.0, "beta_1": 0.949999988079071, "beta_2": 0.9990000128746033, "epsilon": 1e-08, "amsgrad": false}}}}
π

kernel
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Σ
_tf_keras_layerΉ{"class_name": "QuantConv2D", "name": "quant_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 130, 20, 1], "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
°
axis
	gamma
beta
 moving_mean
!moving_variance
"regularization_losses
#	variables
$trainable_variables
%	keras_api
__call__
+&call_and_return_all_conditional_losses"Ϊ
_tf_keras_layerΐ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 56}}}}
ϋ
&regularization_losses
'	variables
(trainable_variables
)	keras_api
__call__
+&call_and_return_all_conditional_losses"κ
_tf_keras_layerΠ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ύ

*kernel
+regularization_losses
,	variables
-trainable_variables
.	keras_api
__call__
+&call_and_return_all_conditional_losses"‘
_tf_keras_layer{"class_name": "QuantConv2D", "name": "quant_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 56}}}}
΄
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4regularization_losses
5	variables
6trainable_variables
7	keras_api
__call__
+&call_and_return_all_conditional_losses"ή
_tf_keras_layerΔ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 56}}}}
?
8regularization_losses
9	variables
:trainable_variables
;	keras_api
__call__
+&call_and_return_all_conditional_losses"ξ
_tf_keras_layerΤ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ύ

<kernel
=regularization_losses
>	variables
?trainable_variables
@	keras_api
__call__
+&call_and_return_all_conditional_losses"‘
_tf_keras_layer{"class_name": "QuantConv2D", "name": "quant_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 56}}}}
΄
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
__call__
+&call_and_return_all_conditional_losses"ή
_tf_keras_layerΔ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 56}}}}
?
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
__call__
+&call_and_return_all_conditional_losses"ξ
_tf_keras_layerΤ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
―

Rkernel
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
 __call__
+‘&call_and_return_all_conditional_losses"
_tf_keras_layerψ{"class_name": "QuantDense", "name": "quant_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "input_quantizer": null, "kernel_quantizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1792}}}}
΅
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\regularization_losses
]	variables
^trainable_variables
_	keras_api
’__call__
+£&call_and_return_all_conditional_losses"ί
_tf_keras_layerΕ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 384}}}}
²

`kernel
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
€__call__
+₯&call_and_return_all_conditional_losses"
_tf_keras_layerϋ{"class_name": "QuantDense", "name": "quant_dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "input_quantizer": null, "kernel_quantizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}}
΅
eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"ί
_tf_keras_layerΕ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 384}}}}
°

nkernel
oregularization_losses
p	variables
qtrainable_variables
r	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layerω{"class_name": "QuantDense", "name": "quant_dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "input_quantizer": null, "kernel_quantizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}}
³
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
ͺ__call__
+«&call_and_return_all_conditional_losses"έ
_tf_keras_layerΓ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 8}}}}
 
|regularization_losses
}	variables
~trainable_variables
	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layerυ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}
ΐ
	iter
beta_1
beta_2

decay
learning_ratemεmζmη*mθ0mι1mκ<mλBmμCmνRmξXmοYmπ`mρfmςgmσnmτtmυumφvχvψvω*vϊ0vϋ1vό<vύBvώCv?RvXvYv`vfvgvnvtvuv"
	optimizer
 "
trackable_list_wrapper

0
1
2
 3
!4
*5
06
17
28
39
<10
B11
C12
D13
E14
R15
X16
Y17
Z18
[19
`20
f21
g22
h23
i24
n25
t26
u27
v28
w29"
trackable_list_wrapper
¦
0
1
2
*3
04
15
<6
B7
C8
R9
X10
Y11
`12
f13
g14
n15
t16
u17"
trackable_list_wrapper
Ώ
regularization_losses
	variables
 layer_regularization_losses
trainable_variables
non_trainable_variables
metrics
layers
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
-:+82quant_conv2d/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
‘
regularization_losses
	variables
 layer_regularization_losses
trainable_variables
non_trainable_variables
metrics
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%82batch_normalization/gamma
&:$82batch_normalization/beta
/:-8 (2batch_normalization/moving_mean
3:18 (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
‘
"regularization_losses
#	variables
 layer_regularization_losses
$trainable_variables
non_trainable_variables
metrics
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
&regularization_losses
'	variables
 layer_regularization_losses
(trainable_variables
non_trainable_variables
metrics
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-882quant_conv2d_1/kernel
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
‘
+regularization_losses
,	variables
 layer_regularization_losses
-trainable_variables
non_trainable_variables
metrics
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'82batch_normalization_1/gamma
(:&82batch_normalization_1/beta
1:/8 (2!batch_normalization_1/moving_mean
5:38 (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
‘
4regularization_losses
5	variables
 layer_regularization_losses
6trainable_variables
non_trainable_variables
metrics
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
8regularization_losses
9	variables
 layer_regularization_losses
:trainable_variables
non_trainable_variables
metrics
 layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-882quant_conv2d_2/kernel
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
‘
=regularization_losses
>	variables
 ‘layer_regularization_losses
?trainable_variables
’non_trainable_variables
£metrics
€layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'82batch_normalization_2/gamma
(:&82batch_normalization_2/beta
1:/8 (2!batch_normalization_2/moving_mean
5:38 (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
‘
Fregularization_losses
G	variables
 ₯layer_regularization_losses
Htrainable_variables
¦non_trainable_variables
§metrics
¨layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
Jregularization_losses
K	variables
 ©layer_regularization_losses
Ltrainable_variables
ͺnon_trainable_variables
«metrics
¬layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
Nregularization_losses
O	variables
 ­layer_regularization_losses
Ptrainable_variables
?non_trainable_variables
―metrics
°layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$
2quant_dense/kernel
 "
trackable_list_wrapper
'
R0"
trackable_list_wrapper
'
R0"
trackable_list_wrapper
‘
Sregularization_losses
T	variables
 ±layer_regularization_losses
Utrainable_variables
²non_trainable_variables
³metrics
΄layers
 __call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_3/gamma
):'2batch_normalization_3/beta
2:0 (2!batch_normalization_3/moving_mean
6:4 (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
<
X0
Y1
Z2
[3"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
‘
\regularization_losses
]	variables
 ΅layer_regularization_losses
^trainable_variables
Άnon_trainable_variables
·metrics
Έlayers
’__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
(:&
2quant_dense_1/kernel
 "
trackable_list_wrapper
'
`0"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
‘
aregularization_losses
b	variables
 Ήlayer_regularization_losses
ctrainable_variables
Ίnon_trainable_variables
»metrics
Όlayers
€__call__
+₯&call_and_return_all_conditional_losses
'₯"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_4/gamma
):'2batch_normalization_4/beta
2:0 (2!batch_normalization_4/moving_mean
6:4 (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
<
f0
g1
h2
i3"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
‘
jregularization_losses
k	variables
 ½layer_regularization_losses
ltrainable_variables
Ύnon_trainable_variables
Ώmetrics
ΐlayers
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
':%	2quant_dense_2/kernel
 "
trackable_list_wrapper
'
n0"
trackable_list_wrapper
'
n0"
trackable_list_wrapper
‘
oregularization_losses
p	variables
 Αlayer_regularization_losses
qtrainable_variables
Βnon_trainable_variables
Γmetrics
Δlayers
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
<
t0
u1
v2
w3"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
‘
xregularization_losses
y	variables
 Εlayer_regularization_losses
ztrainable_variables
Ζnon_trainable_variables
Ηmetrics
Θlayers
ͺ__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
‘
|regularization_losses
}	variables
 Ιlayer_regularization_losses
~trainable_variables
Κnon_trainable_variables
Λmetrics
Μlayers
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
v
 0
!1
22
33
D4
E5
Z6
[7
h8
i9
v10
w11"
trackable_list_wrapper
0
Ν0
Ξ1"
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
 "
trackable_list_wrapper
.
 0
!1"
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
.
20
31"
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
.
D0
E1"
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
.
Z0
[1"
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
.
h0
i1"
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
.
v0
w1"
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
£

Οtotal

Πcount
Ρ
_fn_kwargs
?regularization_losses
Σ	variables
Τtrainable_variables
Υ	keras_api
―__call__
+°&call_and_return_all_conditional_losses"ε
_tf_keras_layerΛ{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
Ι

Φtotal

Χcount
Ψ
_fn_kwargs
Ωregularization_losses
Ϊ	variables
Ϋtrainable_variables
ά	keras_api
±__call__
+²&call_and_return_all_conditional_losses"
_tf_keras_layerρ{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Ο0
Π1"
trackable_list_wrapper
 "
trackable_list_wrapper
€
?regularization_losses
Σ	variables
 έlayer_regularization_losses
Τtrainable_variables
ήnon_trainable_variables
ίmetrics
ΰlayers
―__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Φ0
Χ1"
trackable_list_wrapper
 "
trackable_list_wrapper
€
Ωregularization_losses
Ϊ	variables
 αlayer_regularization_losses
Ϋtrainable_variables
βnon_trainable_variables
γmetrics
δlayers
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Ο0
Π1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Φ0
Χ1"
trackable_list_wrapper
 "
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
ϊ2χ
+__inference_sequential_layer_call_fn_175092
+__inference_sequential_layer_call_fn_174235
+__inference_sequential_layer_call_fn_174381
+__inference_sequential_layer_call_fn_175027ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
σ2π
!__inference__wrapped_model_172568Κ
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
annotationsͺ *:’7
52
quant_conv2d_input?????????
ζ2γ
F__inference_sequential_layer_call_and_return_conditional_losses_174962
F__inference_sequential_layer_call_and_return_conditional_losses_174007
F__inference_sequential_layer_call_and_return_conditional_losses_174823
F__inference_sequential_layer_call_and_return_conditional_losses_174088ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
-__inference_quant_conv2d_layer_call_fn_172584Χ
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
annotationsͺ *7’4
2/+???????????????????????????
§2€
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_172576Χ
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
annotationsͺ *7’4
2/+???????????????????????????
2
4__inference_batch_normalization_layer_call_fn_175173
4__inference_batch_normalization_layer_call_fn_175268
4__inference_batch_normalization_layer_call_fn_175186
4__inference_batch_normalization_layer_call_fn_175255΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
ώ2ϋ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_175242
O__inference_batch_normalization_layer_call_and_return_conditional_losses_175160
O__inference_batch_normalization_layer_call_and_return_conditional_losses_175138
O__inference_batch_normalization_layer_call_and_return_conditional_losses_175220΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
.__inference_max_pooling2d_layer_call_fn_172736ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
±2?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_172730ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
/__inference_quant_conv2d_1_layer_call_fn_172752Χ
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
annotationsͺ *7’4
2/+???????????????????????????8
©2¦
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_172744Χ
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
annotationsͺ *7’4
2/+???????????????????????????8
2
6__inference_batch_normalization_1_layer_call_fn_175362
6__inference_batch_normalization_1_layer_call_fn_175349
6__inference_batch_normalization_1_layer_call_fn_175431
6__inference_batch_normalization_1_layer_call_fn_175444΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_175396
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_175418
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_175314
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_175336΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
0__inference_max_pooling2d_1_layer_call_fn_172904ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
³2°
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_172898ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
/__inference_quant_conv2d_2_layer_call_fn_172920Χ
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
annotationsͺ *7’4
2/+???????????????????????????8
©2¦
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_172912Χ
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
annotationsͺ *7’4
2/+???????????????????????????8
2
6__inference_batch_normalization_2_layer_call_fn_175525
6__inference_batch_normalization_2_layer_call_fn_175538
6__inference_batch_normalization_2_layer_call_fn_175607
6__inference_batch_normalization_2_layer_call_fn_175620΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175594
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175512
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175490
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175572΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
2
0__inference_max_pooling2d_2_layer_call_fn_173072ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
³2°
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_173066ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
?2Ο
(__inference_flatten_layer_call_fn_175631’
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
annotationsͺ *
 
ν2κ
C__inference_flatten_layer_call_and_return_conditional_losses_175626’
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
annotationsͺ *
 
Φ2Σ
,__inference_quant_dense_layer_call_fn_175645’
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
annotationsͺ *
 
ρ2ξ
G__inference_quant_dense_layer_call_and_return_conditional_losses_175638’
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
annotationsͺ *
 
ͺ2§
6__inference_batch_normalization_3_layer_call_fn_175756
6__inference_batch_normalization_3_layer_call_fn_175769΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
ΰ2έ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_175720
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_175743΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
Ψ2Υ
.__inference_quant_dense_1_layer_call_fn_175783’
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
annotationsͺ *
 
σ2π
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_175776’
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
annotationsͺ *
 
ͺ2§
6__inference_batch_normalization_4_layer_call_fn_175907
6__inference_batch_normalization_4_layer_call_fn_175894΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
ΰ2έ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_175858
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_175881΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
Ψ2Υ
.__inference_quant_dense_2_layer_call_fn_175921’
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
annotationsͺ *
 
σ2π
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_175914’
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
annotationsͺ *
 
ͺ2§
6__inference_batch_normalization_5_layer_call_fn_176045
6__inference_batch_normalization_5_layer_call_fn_176032΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
ΰ2έ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_175996
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_176019΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
Υ2?
+__inference_activation_layer_call_fn_176055’
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
annotationsͺ *
 
π2ν
F__inference_activation_layer_call_and_return_conditional_losses_176050’
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
annotationsͺ *
 
>B<
$__inference_signature_wrapper_174600quant_conv2d_input
Μ2ΙΖ
½²Ή
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
kwonlydefaultsͺ

trainingp 
annotationsͺ *
 
Μ2ΙΖ
½²Ή
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
kwonlydefaultsͺ

trainingp 
annotationsͺ *
 
Μ2ΙΖ
½²Ή
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
kwonlydefaultsͺ

trainingp 
annotationsͺ *
 
Μ2ΙΖ
½²Ή
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
kwonlydefaultsͺ

trainingp 
annotationsͺ *
 Ε
!__inference__wrapped_model_172568 !*0123<BCDER[XZY`ifhgnwtvuD’A
:’7
52
quant_conv2d_input?????????
ͺ "7ͺ4
2

activation$!

activation?????????’
F__inference_activation_layer_call_and_return_conditional_losses_176050X/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 z
+__inference_activation_layer_call_fn_176055K/’,
%’"
 
inputs?????????
ͺ "?????????Η
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_175314r0123;’8
1’.
(%
inputs?????????A
8
p
ͺ "-’*
# 
0?????????A
8
 Η
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_175336r0123;’8
1’.
(%
inputs?????????A
8
p 
ͺ "-’*
# 
0?????????A
8
 μ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1753960123M’J
C’@
:7
inputs+???????????????????????????8
p
ͺ "?’<
52
0+???????????????????????????8
 μ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1754180123M’J
C’@
:7
inputs+???????????????????????????8
p 
ͺ "?’<
52
0+???????????????????????????8
 
6__inference_batch_normalization_1_layer_call_fn_175349e0123;’8
1’.
(%
inputs?????????A
8
p
ͺ " ?????????A
8
6__inference_batch_normalization_1_layer_call_fn_175362e0123;’8
1’.
(%
inputs?????????A
8
p 
ͺ " ?????????A
8Δ
6__inference_batch_normalization_1_layer_call_fn_1754310123M’J
C’@
:7
inputs+???????????????????????????8
p
ͺ "2/+???????????????????????????8Δ
6__inference_batch_normalization_1_layer_call_fn_1754440123M’J
C’@
:7
inputs+???????????????????????????8
p 
ͺ "2/+???????????????????????????8μ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175490BCDEM’J
C’@
:7
inputs+???????????????????????????8
p
ͺ "?’<
52
0+???????????????????????????8
 μ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175512BCDEM’J
C’@
:7
inputs+???????????????????????????8
p 
ͺ "?’<
52
0+???????????????????????????8
 Η
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175572rBCDE;’8
1’.
(%
inputs????????? 8
p
ͺ "-’*
# 
0????????? 8
 Η
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_175594rBCDE;’8
1’.
(%
inputs????????? 8
p 
ͺ "-’*
# 
0????????? 8
 Δ
6__inference_batch_normalization_2_layer_call_fn_175525BCDEM’J
C’@
:7
inputs+???????????????????????????8
p
ͺ "2/+???????????????????????????8Δ
6__inference_batch_normalization_2_layer_call_fn_175538BCDEM’J
C’@
:7
inputs+???????????????????????????8
p 
ͺ "2/+???????????????????????????8
6__inference_batch_normalization_2_layer_call_fn_175607eBCDE;’8
1’.
(%
inputs????????? 8
p
ͺ " ????????? 8
6__inference_batch_normalization_2_layer_call_fn_175620eBCDE;’8
1’.
(%
inputs????????? 8
p 
ͺ " ????????? 8Ή
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_175720dZ[XY4’1
*’'
!
inputs?????????
p
ͺ "&’#

0?????????
 Ή
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_175743d[XZY4’1
*’'
!
inputs?????????
p 
ͺ "&’#

0?????????
 
6__inference_batch_normalization_3_layer_call_fn_175756WZ[XY4’1
*’'
!
inputs?????????
p
ͺ "?????????
6__inference_batch_normalization_3_layer_call_fn_175769W[XZY4’1
*’'
!
inputs?????????
p 
ͺ "?????????Ή
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_175858dhifg4’1
*’'
!
inputs?????????
p
ͺ "&’#

0?????????
 Ή
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_175881difhg4’1
*’'
!
inputs?????????
p 
ͺ "&’#

0?????????
 
6__inference_batch_normalization_4_layer_call_fn_175894Whifg4’1
*’'
!
inputs?????????
p
ͺ "?????????
6__inference_batch_normalization_4_layer_call_fn_175907Wifhg4’1
*’'
!
inputs?????????
p 
ͺ "?????????·
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_175996bvwtu3’0
)’&
 
inputs?????????
p
ͺ "%’"

0?????????
 ·
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_176019bwtvu3’0
)’&
 
inputs?????????
p 
ͺ "%’"

0?????????
 
6__inference_batch_normalization_5_layer_call_fn_176032Uvwtu3’0
)’&
 
inputs?????????
p
ͺ "?????????
6__inference_batch_normalization_5_layer_call_fn_176045Uwtvu3’0
)’&
 
inputs?????????
p 
ͺ "?????????κ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_175138 !M’J
C’@
:7
inputs+???????????????????????????8
p
ͺ "?’<
52
0+???????????????????????????8
 κ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_175160 !M’J
C’@
:7
inputs+???????????????????????????8
p 
ͺ "?’<
52
0+???????????????????????????8
 Η
O__inference_batch_normalization_layer_call_and_return_conditional_losses_175220t !<’9
2’/
)&
inputs?????????8
p
ͺ ".’+
$!
0?????????8
 Η
O__inference_batch_normalization_layer_call_and_return_conditional_losses_175242t !<’9
2’/
)&
inputs?????????8
p 
ͺ ".’+
$!
0?????????8
 Β
4__inference_batch_normalization_layer_call_fn_175173 !M’J
C’@
:7
inputs+???????????????????????????8
p
ͺ "2/+???????????????????????????8Β
4__inference_batch_normalization_layer_call_fn_175186 !M’J
C’@
:7
inputs+???????????????????????????8
p 
ͺ "2/+???????????????????????????8
4__inference_batch_normalization_layer_call_fn_175255g !<’9
2’/
)&
inputs?????????8
p
ͺ "!?????????8
4__inference_batch_normalization_layer_call_fn_175268g !<’9
2’/
)&
inputs?????????8
p 
ͺ "!?????????8¨
C__inference_flatten_layer_call_and_return_conditional_losses_175626a7’4
-’*
(%
inputs?????????8
ͺ "&’#

0?????????
 
(__inference_flatten_layer_call_fn_175631T7’4
-’*
(%
inputs?????????8
ͺ "?????????ξ
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_172898R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ζ
0__inference_max_pooling2d_1_layer_call_fn_172904R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????ξ
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_173066R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ζ
0__inference_max_pooling2d_2_layer_call_fn_173072R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????μ
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_172730R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Δ
.__inference_max_pooling2d_layer_call_fn_172736R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????ή
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_172744*I’F
?’<
:7
inputs+???????????????????????????8
ͺ "?’<
52
0+???????????????????????????8
 Ά
/__inference_quant_conv2d_1_layer_call_fn_172752*I’F
?’<
:7
inputs+???????????????????????????8
ͺ "2/+???????????????????????????8ή
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_172912<I’F
?’<
:7
inputs+???????????????????????????8
ͺ "?’<
52
0+???????????????????????????8
 Ά
/__inference_quant_conv2d_2_layer_call_fn_172920<I’F
?’<
:7
inputs+???????????????????????????8
ͺ "2/+???????????????????????????8ά
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_172576I’F
?’<
:7
inputs+???????????????????????????
ͺ "?’<
52
0+???????????????????????????8
 ΄
-__inference_quant_conv2d_layer_call_fn_172584I’F
?’<
:7
inputs+???????????????????????????
ͺ "2/+???????????????????????????8ͺ
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_175776]`0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
.__inference_quant_dense_1_layer_call_fn_175783P`0’-
&’#
!
inputs?????????
ͺ "?????????©
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_175914\n0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????
 
.__inference_quant_dense_2_layer_call_fn_175921On0’-
&’#
!
inputs?????????
ͺ "?????????¨
G__inference_quant_dense_layer_call_and_return_conditional_losses_175638]R0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
,__inference_quant_dense_layer_call_fn_175645PR0’-
&’#
!
inputs?????????
ͺ "?????????ΰ
F__inference_sequential_layer_call_and_return_conditional_losses_174007 !*0123<BCDERZ[XY`hifgnvwtuL’I
B’?
52
quant_conv2d_input?????????
p

 
ͺ "%’"

0?????????
 ΰ
F__inference_sequential_layer_call_and_return_conditional_losses_174088 !*0123<BCDER[XZY`ifhgnwtvuL’I
B’?
52
quant_conv2d_input?????????
p 

 
ͺ "%’"

0?????????
 Τ
F__inference_sequential_layer_call_and_return_conditional_losses_174823 !*0123<BCDERZ[XY`hifgnvwtu@’=
6’3
)&
inputs?????????
p

 
ͺ "%’"

0?????????
 Τ
F__inference_sequential_layer_call_and_return_conditional_losses_174962 !*0123<BCDER[XZY`ifhgnwtvu@’=
6’3
)&
inputs?????????
p 

 
ͺ "%’"

0?????????
 Έ
+__inference_sequential_layer_call_fn_174235 !*0123<BCDERZ[XY`hifgnvwtuL’I
B’?
52
quant_conv2d_input?????????
p

 
ͺ "?????????Έ
+__inference_sequential_layer_call_fn_174381 !*0123<BCDER[XZY`ifhgnwtvuL’I
B’?
52
quant_conv2d_input?????????
p 

 
ͺ "?????????«
+__inference_sequential_layer_call_fn_175027| !*0123<BCDERZ[XY`hifgnvwtu@’=
6’3
)&
inputs?????????
p

 
ͺ "?????????«
+__inference_sequential_layer_call_fn_175092| !*0123<BCDER[XZY`ifhgnwtvu@’=
6’3
)&
inputs?????????
p 

 
ͺ "?????????ή
$__inference_signature_wrapper_174600΅ !*0123<BCDER[XZY`ifhgnwtvuZ’W
’ 
PͺM
K
quant_conv2d_input52
quant_conv2d_input?????????"7ͺ4
2

activation$!

activation?????????