Жд'
Ђэ
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
dtypetypeИ
Њ
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02v1.12.1-25073-g2c5e22190c8±њ!
К
quant_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_namequant_conv2d/kernel
Г
'quant_conv2d/kernel/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel*&
_output_shapes
:0*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:0*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:0*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:0*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:0*
dtype0
О
quant_conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*&
shared_namequant_conv2d_1/kernel
З
)quant_conv2d_1/kernel/Read/ReadVariableOpReadVariableOpquant_conv2d_1/kernel*&
_output_shapes
:00*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:0*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:0*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:0*
dtype0
Ґ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:0*
dtype0
О
quant_conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*&
shared_namequant_conv2d_2/kernel
З
)quant_conv2d_2/kernel/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel*&
_output_shapes
:00*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:0*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:0*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:0*
dtype0
Ґ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:0*
dtype0
О
quant_conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*&
shared_namequant_conv2d_3/kernel
З
)quant_conv2d_3/kernel/Read/ReadVariableOpReadVariableOpquant_conv2d_3/kernel*&
_output_shapes
:00*
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:0*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:0*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:0*
dtype0
Ґ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:0*
dtype0
В
quant_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*#
shared_namequant_dense/kernel
{
&quant_dense/kernel/Read/ReadVariableOpReadVariableOpquant_dense/kernel* 
_output_shapes
:
АА*
dtype0
П
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_4/gamma
И
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_4/beta
Ж
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_4/moving_mean
Ф
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:А*
dtype0
£
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_4/moving_variance
Ь
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:А*
dtype0
Е
quant_dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*%
shared_namequant_dense_1/kernel
~
(quant_dense_1/kernel/Read/ReadVariableOpReadVariableOpquant_dense_1/kernel*
_output_shapes
:	А*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
Ґ
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:*
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
Ш
Adam/quant_conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_nameAdam/quant_conv2d/kernel/m
С
.Adam/quant_conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d/kernel/m*&
_output_shapes
:0*
dtype0
Ш
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" Adam/batch_normalization/gamma/m
С
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:0*
dtype0
Ц
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!Adam/batch_normalization/beta/m
П
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:0*
dtype0
Ь
Adam/quant_conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*-
shared_nameAdam/quant_conv2d_1/kernel/m
Х
0Adam/quant_conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_1/kernel/m*&
_output_shapes
:00*
dtype0
Ь
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/batch_normalization_1/gamma/m
Х
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:0*
dtype0
Ъ
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/batch_normalization_1/beta/m
У
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:0*
dtype0
Ь
Adam/quant_conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*-
shared_nameAdam/quant_conv2d_2/kernel/m
Х
0Adam/quant_conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_2/kernel/m*&
_output_shapes
:00*
dtype0
Ь
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/batch_normalization_2/gamma/m
Х
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:0*
dtype0
Ъ
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/batch_normalization_2/beta/m
У
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:0*
dtype0
Ь
Adam/quant_conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*-
shared_nameAdam/quant_conv2d_3/kernel/m
Х
0Adam/quant_conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_3/kernel/m*&
_output_shapes
:00*
dtype0
Ь
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/batch_normalization_3/gamma/m
Х
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
:0*
dtype0
Ъ
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/batch_normalization_3/beta/m
У
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
:0*
dtype0
Р
Adam/quant_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА**
shared_nameAdam/quant_dense/kernel/m
Й
-Adam/quant_dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_dense/kernel/m* 
_output_shapes
:
АА*
dtype0
Э
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_4/gamma/m
Ц
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_4/beta/m
Ф
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes	
:А*
dtype0
У
Adam/quant_dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*,
shared_nameAdam/quant_dense_1/kernel/m
М
/Adam/quant_dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_dense_1/kernel/m*
_output_shapes
:	А*
dtype0
Ь
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/m
Х
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/m
У
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
:*
dtype0
Ш
Adam/quant_conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_nameAdam/quant_conv2d/kernel/v
С
.Adam/quant_conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d/kernel/v*&
_output_shapes
:0*
dtype0
Ш
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" Adam/batch_normalization/gamma/v
С
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:0*
dtype0
Ц
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*0
shared_name!Adam/batch_normalization/beta/v
П
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:0*
dtype0
Ь
Adam/quant_conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*-
shared_nameAdam/quant_conv2d_1/kernel/v
Х
0Adam/quant_conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_1/kernel/v*&
_output_shapes
:00*
dtype0
Ь
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/batch_normalization_1/gamma/v
Х
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:0*
dtype0
Ъ
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/batch_normalization_1/beta/v
У
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:0*
dtype0
Ь
Adam/quant_conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*-
shared_nameAdam/quant_conv2d_2/kernel/v
Х
0Adam/quant_conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_2/kernel/v*&
_output_shapes
:00*
dtype0
Ь
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/batch_normalization_2/gamma/v
Х
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:0*
dtype0
Ъ
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/batch_normalization_2/beta/v
У
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:0*
dtype0
Ь
Adam/quant_conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*-
shared_nameAdam/quant_conv2d_3/kernel/v
Х
0Adam/quant_conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_3/kernel/v*&
_output_shapes
:00*
dtype0
Ь
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*3
shared_name$"Adam/batch_normalization_3/gamma/v
Х
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
:0*
dtype0
Ъ
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!Adam/batch_normalization_3/beta/v
У
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
:0*
dtype0
Р
Adam/quant_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА**
shared_nameAdam/quant_dense/kernel/v
Й
-Adam/quant_dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_dense/kernel/v* 
_output_shapes
:
АА*
dtype0
Э
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_4/gamma/v
Ц
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_4/beta/v
Ф
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes	
:А*
dtype0
У
Adam/quant_dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*,
shared_nameAdam/quant_dense_1/kernel/v
М
/Adam/quant_dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_dense_1/kernel/v*
_output_shapes
:	А*
dtype0
Ь
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/v
Х
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/v
У
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
°°
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*џ†
value–†Bћ† Bƒ†
Э
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
t
kernel_quantizer

kernel
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
 regularization_losses
!trainable_variables
"	keras_api
Ч
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(	variables
)regularization_losses
*trainable_variables
+	keras_api
Й
,kernel_quantizer
-input_quantizer

.kernel
/	variables
0regularization_losses
1trainable_variables
2	keras_api
R
3	variables
4regularization_losses
5trainable_variables
6	keras_api
Ч
7axis
	8gamma
9beta
:moving_mean
;moving_variance
<	variables
=regularization_losses
>trainable_variables
?	keras_api
Й
@kernel_quantizer
Ainput_quantizer

Bkernel
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
R
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
Ч
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
Й
Tkernel_quantizer
Uinput_quantizer

Vkernel
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
R
[	variables
\regularization_losses
]trainable_variables
^	keras_api
Ч
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
R
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
Й
lkernel_quantizer
minput_quantizer

nkernel
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
Ч
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
М
|kernel_quantizer
}input_quantizer

~kernel
	variables
Аregularization_losses
Бtrainable_variables
В	keras_api
†
	Гaxis

Дgamma
	Еbeta
Жmoving_mean
Зmoving_variance
И	variables
Йregularization_losses
Кtrainable_variables
Л	keras_api
V
М	variables
Нregularization_losses
Оtrainable_variables
П	keras_api
±
	Рiter
Сbeta_1
Тbeta_2

Уdecay
Фlearning_ratem„$mЎ%mў.mЏ8mџ9m№BmЁLmёMmяVmа`mбamвnmгtmдumе~mж	Дmз	Еmиvй$vк%vл.vм8vн9vоBvпLvрMvсVvт`vуavфnvхtvцuvч~vш	Дvщ	Еvъ
к
0
$1
%2
&3
'4
.5
86
97
:8
;9
B10
L11
M12
N13
O14
V15
`16
a17
b18
c19
n20
t21
u22
v23
w24
~25
Д26
Е27
Ж28
З29
 
И
0
$1
%2
.3
84
95
B6
L7
M8
V9
`10
a11
n12
t13
u14
~15
Д16
Е17
Ю
Хmetrics
Цlayers
 Чlayer_regularization_losses
Шnon_trainable_variables
	variables
regularization_losses
trainable_variables
 
l
Щ_custom_metrics
Ъ	variables
Ыregularization_losses
Ьtrainable_variables
Э	keras_api
_]
VARIABLE_VALUEquant_conv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
Ю
Юmetrics
Яlayers
 †layer_regularization_losses
°non_trainable_variables
	variables
regularization_losses
trainable_variables
 
 
 
Ю
Ґmetrics
£layers
 §layer_regularization_losses
•non_trainable_variables
	variables
 regularization_losses
!trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
&2
'3
 

$0
%1
Ю
¶metrics
Іlayers
 ®layer_regularization_losses
©non_trainable_variables
(	variables
)regularization_losses
*trainable_variables
l
™_custom_metrics
Ђ	variables
ђregularization_losses
≠trainable_variables
Ѓ	keras_api
V
ѓ	variables
∞regularization_losses
±trainable_variables
≤	keras_api
a_
VARIABLE_VALUEquant_conv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

.0
 

.0
Ю
≥metrics
іlayers
 µlayer_regularization_losses
ґnon_trainable_variables
/	variables
0regularization_losses
1trainable_variables
 
 
 
Ю
Јmetrics
Єlayers
 єlayer_regularization_losses
Їnon_trainable_variables
3	variables
4regularization_losses
5trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

80
91
:2
;3
 

80
91
Ю
їmetrics
Љlayers
 љlayer_regularization_losses
Њnon_trainable_variables
<	variables
=regularization_losses
>trainable_variables
l
њ_custom_metrics
ј	variables
Ѕregularization_losses
¬trainable_variables
√	keras_api
V
ƒ	variables
≈regularization_losses
∆trainable_variables
«	keras_api
a_
VARIABLE_VALUEquant_conv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

B0
 

B0
Ю
»metrics
…layers
  layer_regularization_losses
Ћnon_trainable_variables
C	variables
Dregularization_losses
Etrainable_variables
 
 
 
Ю
ћmetrics
Ќlayers
 ќlayer_regularization_losses
ѕnon_trainable_variables
G	variables
Hregularization_losses
Itrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
N2
O3
 

L0
M1
Ю
–metrics
—layers
 “layer_regularization_losses
”non_trainable_variables
P	variables
Qregularization_losses
Rtrainable_variables
l
‘_custom_metrics
’	variables
÷regularization_losses
„trainable_variables
Ў	keras_api
V
ў	variables
Џregularization_losses
џtrainable_variables
№	keras_api
a_
VARIABLE_VALUEquant_conv2d_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE

V0
 

V0
Ю
Ёmetrics
ёlayers
 яlayer_regularization_losses
аnon_trainable_variables
W	variables
Xregularization_losses
Ytrainable_variables
 
 
 
Ю
бmetrics
вlayers
 гlayer_regularization_losses
дnon_trainable_variables
[	variables
\regularization_losses
]trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
b2
c3
 

`0
a1
Ю
еmetrics
жlayers
 зlayer_regularization_losses
иnon_trainable_variables
d	variables
eregularization_losses
ftrainable_variables
 
 
 
Ю
йmetrics
кlayers
 лlayer_regularization_losses
мnon_trainable_variables
h	variables
iregularization_losses
jtrainable_variables
l
н_custom_metrics
о	variables
пregularization_losses
рtrainable_variables
с	keras_api
V
т	variables
уregularization_losses
фtrainable_variables
х	keras_api
^\
VARIABLE_VALUEquant_dense/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE

n0
 

n0
Ю
цmetrics
чlayers
 шlayer_regularization_losses
щnon_trainable_variables
o	variables
pregularization_losses
qtrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

t0
u1
v2
w3
 

t0
u1
Ю
ъmetrics
ыlayers
 ьlayer_regularization_losses
эnon_trainable_variables
x	variables
yregularization_losses
ztrainable_variables
l
ю_custom_metrics
€	variables
Аregularization_losses
Бtrainable_variables
В	keras_api
V
Г	variables
Дregularization_losses
Еtrainable_variables
Ж	keras_api
a_
VARIABLE_VALUEquant_dense_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE

~0
 

~0
†
Зmetrics
Иlayers
 Йlayer_regularization_losses
Кnon_trainable_variables
	variables
Аregularization_losses
Бtrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Д0
Е1
Ж2
З3
 

Д0
Е1
°
Лmetrics
Мlayers
 Нlayer_regularization_losses
Оnon_trainable_variables
И	variables
Йregularization_losses
Кtrainable_variables
 
 
 
°
Пmetrics
Рlayers
 Сlayer_regularization_losses
Тnon_trainable_variables
М	variables
Нregularization_losses
Оtrainable_variables
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

У0
Ф1
Ж
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
17
 
X
&0
'1
:2
;3
N4
O5
b6
c7
v8
w9
Ж10
З11
 
 
 
 
°
Хmetrics
Цlayers
 Чlayer_regularization_losses
Шnon_trainable_variables
Ъ	variables
Ыregularization_losses
Ьtrainable_variables
 

0
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
&0
'1
 
 
 
 
°
Щmetrics
Ъlayers
 Ыlayer_regularization_losses
Ьnon_trainable_variables
Ђ	variables
ђregularization_losses
≠trainable_variables
 
 
 
°
Эmetrics
Юlayers
 Яlayer_regularization_losses
†non_trainable_variables
ѓ	variables
∞regularization_losses
±trainable_variables
 

,0
-1
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
:0
;1
 
 
 
 
°
°metrics
Ґlayers
 £layer_regularization_losses
§non_trainable_variables
ј	variables
Ѕregularization_losses
¬trainable_variables
 
 
 
°
•metrics
¶layers
 Іlayer_regularization_losses
®non_trainable_variables
ƒ	variables
≈regularization_losses
∆trainable_variables
 

@0
A1
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
N0
O1
 
 
 
 
°
©metrics
™layers
 Ђlayer_regularization_losses
ђnon_trainable_variables
’	variables
÷regularization_losses
„trainable_variables
 
 
 
°
≠metrics
Ѓlayers
 ѓlayer_regularization_losses
∞non_trainable_variables
ў	variables
Џregularization_losses
џtrainable_variables
 

T0
U1
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
b0
c1
 
 
 
 
 
 
 
 
°
±metrics
≤layers
 ≥layer_regularization_losses
іnon_trainable_variables
о	variables
пregularization_losses
рtrainable_variables
 
 
 
°
µmetrics
ґlayers
 Јlayer_regularization_losses
Єnon_trainable_variables
т	variables
уregularization_losses
фtrainable_variables
 

l0
m1
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
°
єmetrics
Їlayers
 їlayer_regularization_losses
Љnon_trainable_variables
€	variables
Аregularization_losses
Бtrainable_variables
 
 
 
°
љmetrics
Њlayers
 њlayer_regularization_losses
јnon_trainable_variables
Г	variables
Дregularization_losses
Еtrainable_variables
 

|0
}1
 
 
 
 
 

Ж0
З1
 
 
 
 


Ѕtotal

¬count
√
_fn_kwargs
ƒ	variables
≈regularization_losses
∆trainable_variables
«	keras_api


»total

…count
 
_fn_kwargs
Ћ	variables
ћregularization_losses
Ќtrainable_variables
ќ	keras_api
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

Ѕ0
¬1
 
 
°
ѕmetrics
–layers
 —layer_regularization_losses
“non_trainable_variables
ƒ	variables
≈regularization_losses
∆trainable_variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

»0
…1
 
 
°
”metrics
‘layers
 ’layer_regularization_losses
÷non_trainable_variables
Ћ	variables
ћregularization_losses
Ќtrainable_variables
 
 
 

Ѕ0
¬1
 
 
 

»0
…1
ГА
VARIABLE_VALUEAdam/quant_conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/quant_conv2d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/quant_conv2d_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/quant_conv2d_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/quant_dense/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/quant_dense_1/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/quant_conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/quant_conv2d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/quant_conv2d_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/quant_conv2d_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/quant_dense/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/quant_dense_1/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ч
"serving_default_quant_conv2d_inputPlaceholder*0
_output_shapes
:€€€€€€€€€В*
dtype0*%
shape:€€€€€€€€€В
¶	
StatefulPartitionedCallStatefulPartitionedCall"serving_default_quant_conv2d_inputquant_conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancequant_conv2d_1/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancequant_conv2d_2/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancequant_conv2d_3/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancequant_dense/kernel%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betaquant_dense_1/kernel%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/beta**
Tin#
!2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_203105
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ь
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'quant_conv2d/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp)quant_conv2d_1/kernel/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp)quant_conv2d_2/kernel/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp)quant_conv2d_3/kernel/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp&quant_dense/kernel/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp(quant_dense_1/kernel/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/quant_conv2d/kernel/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp0Adam/quant_conv2d_1/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp0Adam/quant_conv2d_2/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp0Adam/quant_conv2d_3/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp-Adam/quant_dense/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp/Adam/quant_dense_1/kernel/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp.Adam/quant_conv2d/kernel/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp0Adam/quant_conv2d_1/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp0Adam/quant_conv2d_2/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp0Adam/quant_conv2d_3/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp-Adam/quant_dense/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp/Adam/quant_dense_1/kernel/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpConst*X
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
__inference__traced_save_205170
ї
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamequant_conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancequant_conv2d_1/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancequant_conv2d_2/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancequant_conv2d_3/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancequant_dense/kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancequant_dense_1/kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/quant_conv2d/kernel/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/quant_conv2d_1/kernel/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/quant_conv2d_2/kernel/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/quant_conv2d_3/kernel/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/quant_dense/kernel/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/quant_dense_1/kernel/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/quant_conv2d/kernel/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/quant_conv2d_1/kernel/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/quant_conv2d_2/kernel/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/quant_conv2d_3/kernel/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/quant_dense/kernel/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/quant_dense_1/kernel/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/v*W
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
"__inference__traced_restore_205407УЈ
у%
Ъ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_200971

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_200956
assignmovingavg_1_200963
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/200956*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/200956*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_200956*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/200956*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/200956*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_200956AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/200956*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/200963*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/200963*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_200963*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/200963*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/200963*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_200963AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/200963*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
Љ
ф
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204271

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constм
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
ѕ
b
D__inference_ste_sign_layer_call_and_return_conditional_losses_200846

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:02
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:02
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:02
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:02

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-200837*8
_output_shapes&
$:0:02
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:02

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
≠%
Ь
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_202087

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_202072
assignmovingavg_1_202079
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/202072*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/202072*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_202072*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/202072*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/202072*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_202072AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/202072*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/202079*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202079*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_202079*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202079*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202079*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_202079AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/202079*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€ 02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€ 0
 
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
—
G
+__inference_ste_sign_5_layer_call_fn_204904

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*&
_output_shapes
:00*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_5_layer_call_and_return_conditional_losses_2014762
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:002

Identity"
identityIdentity:output:0*%
_input_shapes
:00:N J
&
_output_shapes
:00
 
_user_specified_nameinputs
≈	
d
F__inference_ste_sign_6_layer_call_and_return_conditional_losses_204916

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/yy
addAddV2Sign:y:0add/y:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identityв
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-204907*n
_output_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
	IdentityNД

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
Ј
Љ
$__inference_signature_wrapper_203105
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
identityИҐStatefulPartitionedCallы
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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_2008282
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:€€€€€€€€€В
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
ъ
И
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_204766

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOp^
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

LogicalAndТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/add_1џ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
Њ
G
+__inference_ste_sign_6_layer_call_fn_204921

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_6_layer_call_and_return_conditional_losses_2014532
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
О
•
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_201277

inputs"
conv2d_readvariableop_resource
identityИҐConv2D/ReadVariableOp≥
ste_sign_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_2012432
ste_sign_4/PartitionedCallІ
ste_sign_4/IdentityIdentity#ste_sign_4/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
ste_sign_4/IdentityХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOpљ
!Conv2D/ste_sign_3/PartitionedCallPartitionedCallConv2D/ReadVariableOp:value:0*
Tin
2*
Tout
2*&
_output_shapes
:00*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_2012662#
!Conv2D/ste_sign_3/PartitionedCall°
Conv2D/ste_sign_3/IdentityIdentity*Conv2D/ste_sign_3/PartitionedCall:output:0*
T0*&
_output_shapes
:002
Conv2D/ste_sign_3/Identity—
Conv2DConv2Dste_sign_4/Identity:output:0#Conv2D/ste_sign_3/Identity:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*
paddingSAME*
strides
2
Conv2DХ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs:

_output_shapes
: 
Д
І
4__inference_batch_normalization_layer_call_fn_203932

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2019922
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€A
02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€A
0
 
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
њ
Ј
+__inference_sequential_layer_call_fn_203769

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
identityИҐStatefulPartitionedCallФ
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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2028172
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€В
 
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
м
L
0__inference_max_pooling2d_3_layer_call_fn_201507

inputs
identityЂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2015012
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≠%
Ь
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_204073

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_204058
assignmovingavg_1_204065
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/204058*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/204058*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_204058*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/204058*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/204058*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_204058AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/204058*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/204065*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204065*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_204065*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204065*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204065*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_204065AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/204065*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€ 02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€ 0
 
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
х%
Ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204343

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_204328
assignmovingavg_1_204335
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/204328*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/204328*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_204328*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/204328*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/204328*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_204328AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/204328*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/204335*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204335*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_204335*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204335*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204335*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_204335AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/204335*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
Ќ
E
)__inference_ste_sign_layer_call_fn_204819

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*&
_output_shapes
:0*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_ste_sign_layer_call_and_return_conditional_losses_2008462
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:02

Identity"
identityIdentity:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
г
√
+__inference_sequential_layer_call_fn_202733
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
identityИҐStatefulPartitionedCall†
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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2026702
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:€€€€€€€€€В
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
ф
ф
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_204095

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstЏ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€ 02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€ 0
 
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
ф
ф
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204189

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstЏ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€0
 
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
’0
»
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_201752

inputs
assignmovingavg_201727
assignmovingavg_1_201733)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOp^
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

LogicalAndК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradient•
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≥
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/201727*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_201727*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpƒ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/201727*
_output_shapes	
:А2
AssignMovingAvg/subї
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/201727*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_201727AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/201727*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/201733*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_201733*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpќ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201733*
_output_shapes	
:А2
AssignMovingAvg_1/sub≈
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201733*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_201733AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/201733*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/add_1і
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
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
–
©
6__inference_batch_normalization_3_layer_call_fn_204378

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2016012
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
∆»
а,
"__inference__traced_restore_205407
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
9assignvariableop_14_batch_normalization_2_moving_variance-
)assignvariableop_15_quant_conv2d_3_kernel3
/assignvariableop_16_batch_normalization_3_gamma2
.assignvariableop_17_batch_normalization_3_beta9
5assignvariableop_18_batch_normalization_3_moving_mean=
9assignvariableop_19_batch_normalization_3_moving_variance*
&assignvariableop_20_quant_dense_kernel3
/assignvariableop_21_batch_normalization_4_gamma2
.assignvariableop_22_batch_normalization_4_beta9
5assignvariableop_23_batch_normalization_4_moving_mean=
9assignvariableop_24_batch_normalization_4_moving_variance,
(assignvariableop_25_quant_dense_1_kernel3
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
5assignvariableop_47_adam_batch_normalization_2_beta_m4
0assignvariableop_48_adam_quant_conv2d_3_kernel_m:
6assignvariableop_49_adam_batch_normalization_3_gamma_m9
5assignvariableop_50_adam_batch_normalization_3_beta_m1
-assignvariableop_51_adam_quant_dense_kernel_m:
6assignvariableop_52_adam_batch_normalization_4_gamma_m9
5assignvariableop_53_adam_batch_normalization_4_beta_m3
/assignvariableop_54_adam_quant_dense_1_kernel_m:
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
5assignvariableop_65_adam_batch_normalization_2_beta_v4
0assignvariableop_66_adam_quant_conv2d_3_kernel_v:
6assignvariableop_67_adam_batch_normalization_3_gamma_v9
5assignvariableop_68_adam_batch_normalization_3_beta_v1
-assignvariableop_69_adam_quant_dense_kernel_v:
6assignvariableop_70_adam_batch_normalization_4_gamma_v9
5assignvariableop_71_adam_batch_normalization_4_beta_v3
/assignvariableop_72_adam_quant_dense_1_kernel_v:
6assignvariableop_73_adam_batch_normalization_5_gamma_v9
5assignvariableop_74_adam_batch_normalization_5_beta_v
identity_76ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1ѕ)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*џ(
value—(Bќ(KB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesІ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ђ
value°BЮKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices•
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¬
_output_shapesѓ
ђ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityФ
AssignVariableOpAssignVariableOp$assignvariableop_quant_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ґ
AssignVariableOp_1AssignVariableOp,assignvariableop_1_batch_normalization_gammaIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2°
AssignVariableOp_2AssignVariableOp+assignvariableop_2_batch_normalization_betaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3®
AssignVariableOp_3AssignVariableOp2assignvariableop_3_batch_normalization_moving_meanIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4ђ
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_moving_varianceIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ю
AssignVariableOp_5AssignVariableOp(assignvariableop_5_quant_conv2d_1_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6§
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

Identity_8™
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ѓ
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Ґ
AssignVariableOp_10AssignVariableOp)assignvariableop_10_quant_conv2d_2_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11®
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_2_gammaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12І
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_2_betaIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ѓ
AssignVariableOp_13AssignVariableOp5assignvariableop_13_batch_normalization_2_moving_meanIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14≤
AssignVariableOp_14AssignVariableOp9assignvariableop_14_batch_normalization_2_moving_varianceIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ґ
AssignVariableOp_15AssignVariableOp)assignvariableop_15_quant_conv2d_3_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16®
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_3_gammaIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17І
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_3_betaIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ѓ
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_3_moving_meanIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19≤
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_3_moving_varianceIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Я
AssignVariableOp_20AssignVariableOp&assignvariableop_20_quant_dense_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_4_gammaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22І
AssignVariableOp_22AssignVariableOp.assignvariableop_22_batch_normalization_4_betaIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Ѓ
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_4_moving_meanIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24≤
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_4_moving_varianceIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25°
AssignVariableOp_25AssignVariableOp(assignvariableop_25_quant_dense_1_kernelIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26®
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_5_gammaIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27І
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_5_betaIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ѓ
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_5_moving_meanIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29≤
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_5_moving_varianceIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0	*
_output_shapes
:2
Identity_30Ц
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_iterIdentity_30:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Ш
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_beta_1Identity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Ш
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_beta_2Identity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ч
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_decayIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Я
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_learning_rateIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Т
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Т
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37Ф
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_1Identity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38Ф
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_1Identity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39І
AssignVariableOp_39AssignVariableOp.assignvariableop_39_adam_quant_conv2d_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40≠
AssignVariableOp_40AssignVariableOp4assignvariableop_40_adam_batch_normalization_gamma_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41ђ
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
Identity_43ѓ
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_1_gamma_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44Ѓ
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
Identity_46ѓ
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_2_gamma_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47Ѓ
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adam_batch_normalization_2_beta_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48©
AssignVariableOp_48AssignVariableOp0assignvariableop_48_adam_quant_conv2d_3_kernel_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49ѓ
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_3_gamma_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50Ѓ
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_3_beta_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51¶
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_quant_dense_kernel_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52ѓ
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_batch_normalization_4_gamma_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53Ѓ
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adam_batch_normalization_4_beta_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54®
AssignVariableOp_54AssignVariableOp/assignvariableop_54_adam_quant_dense_1_kernel_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55ѓ
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_5_gamma_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56Ѓ
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_batch_normalization_5_beta_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57І
AssignVariableOp_57AssignVariableOp.assignvariableop_57_adam_quant_conv2d_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58≠
AssignVariableOp_58AssignVariableOp4assignvariableop_58_adam_batch_normalization_gamma_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59ђ
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
Identity_61ѓ
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_1_gamma_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62Ѓ
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
Identity_64ѓ
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_2_gamma_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65Ѓ
AssignVariableOp_65AssignVariableOp5assignvariableop_65_adam_batch_normalization_2_beta_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66©
AssignVariableOp_66AssignVariableOp0assignvariableop_66_adam_quant_conv2d_3_kernel_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67ѓ
AssignVariableOp_67AssignVariableOp6assignvariableop_67_adam_batch_normalization_3_gamma_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68Ѓ
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_batch_normalization_3_beta_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69¶
AssignVariableOp_69AssignVariableOp-assignvariableop_69_adam_quant_dense_kernel_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70ѓ
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_4_gamma_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71Ѓ
AssignVariableOp_71AssignVariableOp5assignvariableop_71_adam_batch_normalization_4_beta_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72®
AssignVariableOp_72AssignVariableOp/assignvariableop_72_adam_quant_dense_1_kernel_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73ѓ
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_batch_normalization_5_gamma_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74Ѓ
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_batch_normalization_5_beta_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
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
NoOp–
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75Ё
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*√
_input_shapes±
Ѓ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
’0
»
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_204589

inputs
assignmovingavg_204564
assignmovingavg_1_204570)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOp^
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

LogicalAndК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradient•
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≥
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/204564*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_204564*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpƒ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/204564*
_output_shapes	
:А2
AssignMovingAvg/subї
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/204564*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_204564AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/204564*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/204570*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_204570*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpќ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204570*
_output_shapes	
:А2
AssignMovingAvg_1/sub≈
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204570*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_204570AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/204570*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/add_1і
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
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
Њ
G
+__inference_ste_sign_2_layer_call_fn_204853

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_2010332
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
ћ
І
4__inference_batch_normalization_layer_call_fn_203863

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2010062
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
Ї
т
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203837

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constм
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
И
©
6__inference_batch_normalization_3_layer_call_fn_204473

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2022992
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€0
 
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
љ0
»
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_204743

inputs
assignmovingavg_204718
assignmovingavg_1_204724)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOp^
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

LogicalAndК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≤
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/204718*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_204718*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp√
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/204718*
_output_shapes
:2
AssignMovingAvg/subЇ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/204718*
_output_shapes
:2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_204718AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/204718*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/204724*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_204724*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЌ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204724*
_output_shapes
:2
AssignMovingAvg_1/subƒ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204724*
_output_shapes
:2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_204724AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/204724*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/add_1≥
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
≈	
d
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_201243

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/yy
addAddV2Sign:y:0add/y:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identityв
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-201234*n
_output_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
	IdentityNД

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
И
©
6__inference_batch_normalization_2_layer_call_fn_204202

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2021822
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€0
 
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
–
©
6__inference_batch_normalization_2_layer_call_fn_204297

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2014262
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
И
©
6__inference_batch_normalization_1_layer_call_fn_204108

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2020872
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 0
 
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
—
d
F__inference_ste_sign_5_layer_call_and_return_conditional_losses_201476

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:002
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:002
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:002
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:002

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-201467*8
_output_shapes&
$:00:002
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:002

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:00:N J
&
_output_shapes
:00
 
_user_specified_nameinputs
ъ
И
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_201940

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOp^
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

LogicalAndТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/add_1џ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
и
©
6__inference_batch_normalization_5_layer_call_fn_204779

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2019042
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
Љ
ф
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_201426

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constм
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
—
d
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_204831

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:002
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:002
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:002
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:002

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-204822*8
_output_shapes&
$:00:002
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:002

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:00:N J
&
_output_shapes
:00
 
_user_specified_nameinputs
Кі
≈
F__inference_sequential_layer_call_and_return_conditional_losses_203639

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
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource1
-quant_conv2d_3_conv2d_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource.
*quant_dense_matmul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource0
,quant_dense_1_matmul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource?
;batch_normalization_5_batchnorm_mul_readvariableop_resource=
9batch_normalization_5_batchnorm_readvariableop_1_resource=
9batch_normalization_5_batchnorm_readvariableop_2_resource
identityИҐ3batch_normalization/FusedBatchNormV3/ReadVariableOpҐ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_3/ReadVariableOpҐ&batch_normalization_3/ReadVariableOp_1Ґ.batch_normalization_4/batchnorm/ReadVariableOpҐ0batch_normalization_4/batchnorm/ReadVariableOp_1Ґ0batch_normalization_4/batchnorm/ReadVariableOp_2Ґ2batch_normalization_4/batchnorm/mul/ReadVariableOpҐ.batch_normalization_5/batchnorm/ReadVariableOpҐ0batch_normalization_5/batchnorm/ReadVariableOp_1Ґ0batch_normalization_5/batchnorm/ReadVariableOp_2Ґ2batch_normalization_5/batchnorm/mul/ReadVariableOpҐ"quant_conv2d/Conv2D/ReadVariableOpҐ$quant_conv2d_1/Conv2D/ReadVariableOpҐ$quant_conv2d_2/Conv2D/ReadVariableOpҐ$quant_conv2d_3/Conv2D/ReadVariableOpҐ!quant_dense/MatMul/ReadVariableOpҐ#quant_dense_1/MatMul/ReadVariableOpЉ
"quant_conv2d/Conv2D/ReadVariableOpReadVariableOp+quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02$
"quant_conv2d/Conv2D/ReadVariableOpЂ
!quant_conv2d/Conv2D/ste_sign/SignSign*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:02#
!quant_conv2d/Conv2D/ste_sign/SignН
"quant_conv2d/Conv2D/ste_sign/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2$
"quant_conv2d/Conv2D/ste_sign/add/y“
 quant_conv2d/Conv2D/ste_sign/addAddV2%quant_conv2d/Conv2D/ste_sign/Sign:y:0+quant_conv2d/Conv2D/ste_sign/add/y:output:0*
T0*&
_output_shapes
:02"
 quant_conv2d/Conv2D/ste_sign/add©
#quant_conv2d/Conv2D/ste_sign/Sign_1Sign$quant_conv2d/Conv2D/ste_sign/add:z:0*
T0*&
_output_shapes
:02%
#quant_conv2d/Conv2D/ste_sign/Sign_1і
%quant_conv2d/Conv2D/ste_sign/IdentityIdentity'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*
T0*&
_output_shapes
:02'
%quant_conv2d/Conv2D/ste_sign/IdentityІ
&quant_conv2d/Conv2D/ste_sign/IdentityN	IdentityN'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203417*8
_output_shapes&
$:0:02(
&quant_conv2d/Conv2D/ste_sign/IdentityN–
quant_conv2d/Conv2DConv2Dinputs/quant_conv2d/Conv2D/ste_sign/IdentityN:output:0*
T0*0
_output_shapes
:€€€€€€€€€В0*
paddingSAME*
strides
2
quant_conv2d/Conv2Dƒ
max_pooling2d/MaxPoolMaxPoolquant_conv2d/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€A
0*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolЖ
 batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 batch_normalization/LogicalAnd/xЖ
 batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 batch_normalization/LogicalAnd/yЉ
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2 
batch_normalization/LogicalAnd∞
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:0*
dtype02$
"batch_normalization/ReadVariableOpґ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:0*
dtype02&
$batch_normalization/ReadVariableOp_1г
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpй
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Џ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
0:0:0:0:0:*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3{
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization/Constђ
quant_conv2d_1/ste_sign_2/SignSign(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€A
02 
quant_conv2d_1/ste_sign_2/SignЗ
quant_conv2d_1/ste_sign_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2!
quant_conv2d_1/ste_sign_2/add/yѕ
quant_conv2d_1/ste_sign_2/addAddV2"quant_conv2d_1/ste_sign_2/Sign:y:0(quant_conv2d_1/ste_sign_2/add/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
02
quant_conv2d_1/ste_sign_2/add©
 quant_conv2d_1/ste_sign_2/Sign_1Sign!quant_conv2d_1/ste_sign_2/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€A
02"
 quant_conv2d_1/ste_sign_2/Sign_1і
"quant_conv2d_1/ste_sign_2/IdentityIdentity$quant_conv2d_1/ste_sign_2/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€A
02$
"quant_conv2d_1/ste_sign_2/IdentityЃ
#quant_conv2d_1/ste_sign_2/IdentityN	IdentityN$quant_conv2d_1/ste_sign_2/Sign_1:y:0(batch_normalization/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-203445*J
_output_shapes8
6:€€€€€€€€€A
0:€€€€€€€€€A
02%
#quant_conv2d_1/ste_sign_2/IdentityN¬
$quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02&
$quant_conv2d_1/Conv2D/ReadVariableOpµ
%quant_conv2d_1/Conv2D/ste_sign_1/SignSign,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:002'
%quant_conv2d_1/Conv2D/ste_sign_1/SignХ
&quant_conv2d_1/Conv2D/ste_sign_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2(
&quant_conv2d_1/Conv2D/ste_sign_1/add/yв
$quant_conv2d_1/Conv2D/ste_sign_1/addAddV2)quant_conv2d_1/Conv2D/ste_sign_1/Sign:y:0/quant_conv2d_1/Conv2D/ste_sign_1/add/y:output:0*
T0*&
_output_shapes
:002&
$quant_conv2d_1/Conv2D/ste_sign_1/addµ
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1Sign(quant_conv2d_1/Conv2D/ste_sign_1/add:z:0*
T0*&
_output_shapes
:002)
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1ј
)quant_conv2d_1/Conv2D/ste_sign_1/IdentityIdentity+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0*
T0*&
_output_shapes
:002+
)quant_conv2d_1/Conv2D/ste_sign_1/Identityµ
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityN	IdentityN+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203455*8
_output_shapes&
$:00:002,
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityNэ
quant_conv2d_1/Conv2DConv2D,quant_conv2d_1/ste_sign_2/IdentityN:output:03quant_conv2d_1/Conv2D/ste_sign_1/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
0*
paddingSAME*
strides
2
quant_conv2d_1/Conv2D 
max_pooling2d_1/MaxPoolMaxPoolquant_conv2d_1/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€ 0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolК
"batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_1/LogicalAnd/xК
"batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_1/LogicalAnd/yƒ
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_1/LogicalAndґ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOpЉ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1й
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1и
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 0:0:0:0:0:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_1/ConstЃ
quant_conv2d_2/ste_sign_4/SignSign*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 02 
quant_conv2d_2/ste_sign_4/SignЗ
quant_conv2d_2/ste_sign_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2!
quant_conv2d_2/ste_sign_4/add/yѕ
quant_conv2d_2/ste_sign_4/addAddV2"quant_conv2d_2/ste_sign_4/Sign:y:0(quant_conv2d_2/ste_sign_4/add/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 02
quant_conv2d_2/ste_sign_4/add©
 quant_conv2d_2/ste_sign_4/Sign_1Sign!quant_conv2d_2/ste_sign_4/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€ 02"
 quant_conv2d_2/ste_sign_4/Sign_1і
"quant_conv2d_2/ste_sign_4/IdentityIdentity$quant_conv2d_2/ste_sign_4/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 02$
"quant_conv2d_2/ste_sign_4/Identity∞
#quant_conv2d_2/ste_sign_4/IdentityN	IdentityN$quant_conv2d_2/ste_sign_4/Sign_1:y:0*batch_normalization_1/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-203483*J
_output_shapes8
6:€€€€€€€€€ 0:€€€€€€€€€ 02%
#quant_conv2d_2/ste_sign_4/IdentityN¬
$quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02&
$quant_conv2d_2/Conv2D/ReadVariableOpµ
%quant_conv2d_2/Conv2D/ste_sign_3/SignSign,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:002'
%quant_conv2d_2/Conv2D/ste_sign_3/SignХ
&quant_conv2d_2/Conv2D/ste_sign_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2(
&quant_conv2d_2/Conv2D/ste_sign_3/add/yв
$quant_conv2d_2/Conv2D/ste_sign_3/addAddV2)quant_conv2d_2/Conv2D/ste_sign_3/Sign:y:0/quant_conv2d_2/Conv2D/ste_sign_3/add/y:output:0*
T0*&
_output_shapes
:002&
$quant_conv2d_2/Conv2D/ste_sign_3/addµ
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1Sign(quant_conv2d_2/Conv2D/ste_sign_3/add:z:0*
T0*&
_output_shapes
:002)
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1ј
)quant_conv2d_2/Conv2D/ste_sign_3/IdentityIdentity+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0*
T0*&
_output_shapes
:002+
)quant_conv2d_2/Conv2D/ste_sign_3/Identityµ
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityN	IdentityN+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203493*8
_output_shapes&
$:00:002,
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityNэ
quant_conv2d_2/Conv2DConv2D,quant_conv2d_2/ste_sign_4/IdentityN:output:03quant_conv2d_2/Conv2D/ste_sign_3/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 0*
paddingSAME*
strides
2
quant_conv2d_2/Conv2D 
max_pooling2d_2/MaxPoolMaxPoolquant_conv2d_2/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolК
"batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_2/LogicalAnd/xК
"batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/yƒ
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAndґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1и
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_2/ConstЃ
quant_conv2d_3/ste_sign_6/SignSign*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€02 
quant_conv2d_3/ste_sign_6/SignЗ
quant_conv2d_3/ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2!
quant_conv2d_3/ste_sign_6/add/yѕ
quant_conv2d_3/ste_sign_6/addAddV2"quant_conv2d_3/ste_sign_6/Sign:y:0(quant_conv2d_3/ste_sign_6/add/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€02
quant_conv2d_3/ste_sign_6/add©
 quant_conv2d_3/ste_sign_6/Sign_1Sign!quant_conv2d_3/ste_sign_6/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€02"
 quant_conv2d_3/ste_sign_6/Sign_1і
"quant_conv2d_3/ste_sign_6/IdentityIdentity$quant_conv2d_3/ste_sign_6/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€02$
"quant_conv2d_3/ste_sign_6/Identity∞
#quant_conv2d_3/ste_sign_6/IdentityN	IdentityN$quant_conv2d_3/ste_sign_6/Sign_1:y:0*batch_normalization_2/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-203521*J
_output_shapes8
6:€€€€€€€€€0:€€€€€€€€€02%
#quant_conv2d_3/ste_sign_6/IdentityN¬
$quant_conv2d_3/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02&
$quant_conv2d_3/Conv2D/ReadVariableOpµ
%quant_conv2d_3/Conv2D/ste_sign_5/SignSign,quant_conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:002'
%quant_conv2d_3/Conv2D/ste_sign_5/SignХ
&quant_conv2d_3/Conv2D/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2(
&quant_conv2d_3/Conv2D/ste_sign_5/add/yв
$quant_conv2d_3/Conv2D/ste_sign_5/addAddV2)quant_conv2d_3/Conv2D/ste_sign_5/Sign:y:0/quant_conv2d_3/Conv2D/ste_sign_5/add/y:output:0*
T0*&
_output_shapes
:002&
$quant_conv2d_3/Conv2D/ste_sign_5/addµ
'quant_conv2d_3/Conv2D/ste_sign_5/Sign_1Sign(quant_conv2d_3/Conv2D/ste_sign_5/add:z:0*
T0*&
_output_shapes
:002)
'quant_conv2d_3/Conv2D/ste_sign_5/Sign_1ј
)quant_conv2d_3/Conv2D/ste_sign_5/IdentityIdentity+quant_conv2d_3/Conv2D/ste_sign_5/Sign_1:y:0*
T0*&
_output_shapes
:002+
)quant_conv2d_3/Conv2D/ste_sign_5/Identityµ
*quant_conv2d_3/Conv2D/ste_sign_5/IdentityN	IdentityN+quant_conv2d_3/Conv2D/ste_sign_5/Sign_1:y:0,quant_conv2d_3/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203531*8
_output_shapes&
$:00:002,
*quant_conv2d_3/Conv2D/ste_sign_5/IdentityNэ
quant_conv2d_3/Conv2DConv2D,quant_conv2d_3/ste_sign_6/IdentityN:output:03quant_conv2d_3/Conv2D/ste_sign_5/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€0*
paddingSAME*
strides
2
quant_conv2d_3/Conv2D 
max_pooling2d_3/MaxPoolMaxPoolquant_conv2d_3/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolК
"batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_3/LogicalAnd/xК
"batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/yƒ
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAndґ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_3/ReadVariableOpЉ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_3/ReadVariableOp_1й
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1и
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_3/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_3/Consto
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
flatten/Const§
flatten/ReshapeReshape*batch_normalization_3/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/ReshapeП
quant_dense/ste_sign_8/SignSignflatten/Reshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_8/SignБ
quant_dense/ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
quant_dense/ste_sign_8/add/yЉ
quant_dense/ste_sign_8/addAddV2quant_dense/ste_sign_8/Sign:y:0%quant_dense/ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_8/addЩ
quant_dense/ste_sign_8/Sign_1Signquant_dense/ste_sign_8/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_8/Sign_1§
quant_dense/ste_sign_8/IdentityIdentity!quant_dense/ste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
quant_dense/ste_sign_8/IdentityЗ
 quant_dense/ste_sign_8/IdentityN	IdentityN!quant_dense/ste_sign_8/Sign_1:y:0flatten/Reshape:output:0*
T
2*,
_gradient_op_typeCustomGradient-203561*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2"
 quant_dense/ste_sign_8/IdentityN≥
!quant_dense/MatMul/ReadVariableOpReadVariableOp*quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!quant_dense/MatMul/ReadVariableOp¶
"quant_dense/MatMul/ste_sign_7/SignSign)quant_dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"quant_dense/MatMul/ste_sign_7/SignП
#quant_dense/MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2%
#quant_dense/MatMul/ste_sign_7/add/y–
!quant_dense/MatMul/ste_sign_7/addAddV2&quant_dense/MatMul/ste_sign_7/Sign:y:0,quant_dense/MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
АА2#
!quant_dense/MatMul/ste_sign_7/add¶
$quant_dense/MatMul/ste_sign_7/Sign_1Sign%quant_dense/MatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
АА2&
$quant_dense/MatMul/ste_sign_7/Sign_1±
&quant_dense/MatMul/ste_sign_7/IdentityIdentity(quant_dense/MatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
АА2(
&quant_dense/MatMul/ste_sign_7/IdentityЭ
'quant_dense/MatMul/ste_sign_7/IdentityN	IdentityN(quant_dense/MatMul/ste_sign_7/Sign_1:y:0)quant_dense/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203571*,
_output_shapes
:
АА:
АА2)
'quant_dense/MatMul/ste_sign_7/IdentityN¬
quant_dense/MatMulMatMul)quant_dense/ste_sign_8/IdentityN:output:00quant_dense/MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/MatMulК
"batch_normalization_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_4/LogicalAnd/xК
"batch_normalization_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_4/LogicalAnd/yƒ
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_4/LogicalAnd’
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/yб
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/add¶
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/Rsqrtб
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpё
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/mulѕ
%batch_normalization_4/batchnorm/mul_1Mulquant_dense/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_4/batchnorm/mul_1џ
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1ё
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/mul_2џ
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2№
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/subё
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_4/batchnorm/add_1¶
quant_dense_1/ste_sign_10/SignSign)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
quant_dense_1/ste_sign_10/SignЗ
quant_dense_1/ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2!
quant_dense_1/ste_sign_10/add/y»
quant_dense_1/ste_sign_10/addAddV2"quant_dense_1/ste_sign_10/Sign:y:0(quant_dense_1/ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense_1/ste_sign_10/addҐ
 quant_dense_1/ste_sign_10/Sign_1Sign!quant_dense_1/ste_sign_10/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 quant_dense_1/ste_sign_10/Sign_1≠
"quant_dense_1/ste_sign_10/IdentityIdentity$quant_dense_1/ste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"quant_dense_1/ste_sign_10/Identity°
#quant_dense_1/ste_sign_10/IdentityN	IdentityN$quant_dense_1/ste_sign_10/Sign_1:y:0)batch_normalization_4/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-203599*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2%
#quant_dense_1/ste_sign_10/IdentityNЄ
#quant_dense_1/MatMul/ReadVariableOpReadVariableOp,quant_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02%
#quant_dense_1/MatMul/ReadVariableOpЂ
$quant_dense_1/MatMul/ste_sign_9/SignSign+quant_dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2&
$quant_dense_1/MatMul/ste_sign_9/SignУ
%quant_dense_1/MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2'
%quant_dense_1/MatMul/ste_sign_9/add/y„
#quant_dense_1/MatMul/ste_sign_9/addAddV2(quant_dense_1/MatMul/ste_sign_9/Sign:y:0.quant_dense_1/MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	А2%
#quant_dense_1/MatMul/ste_sign_9/addЂ
&quant_dense_1/MatMul/ste_sign_9/Sign_1Sign'quant_dense_1/MatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	А2(
&quant_dense_1/MatMul/ste_sign_9/Sign_1ґ
(quant_dense_1/MatMul/ste_sign_9/IdentityIdentity*quant_dense_1/MatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	А2*
(quant_dense_1/MatMul/ste_sign_9/Identity£
)quant_dense_1/MatMul/ste_sign_9/IdentityN	IdentityN*quant_dense_1/MatMul/ste_sign_9/Sign_1:y:0+quant_dense_1/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203609**
_output_shapes
:	А:	А2+
)quant_dense_1/MatMul/ste_sign_9/IdentityN 
quant_dense_1/MatMulMatMul,quant_dense_1/ste_sign_10/IdentityN:output:02quant_dense_1/MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
quant_dense_1/MatMulК
"batch_normalization_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_5/LogicalAnd/xК
"batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_5/LogicalAnd/yƒ
 batch_normalization_5/LogicalAnd
LogicalAnd+batch_normalization_5/LogicalAnd/x:output:0+batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_5/LogicalAnd‘
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpУ
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_5/batchnorm/add/yа
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/add•
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtа
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mul–
%batch_normalization_5/batchnorm/mul_1Mulquant_dense_1/MatMul:product:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%batch_normalization_5/batchnorm/mul_1Џ
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1Ё
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2Џ
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2џ
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/subЁ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%batch_normalization_5/batchnorm/add_1Р
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
activation/Softmaxм
IdentityIdentityactivation/Softmax:softmax:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp#^quant_conv2d/Conv2D/ReadVariableOp%^quant_conv2d_1/Conv2D/ReadVariableOp%^quant_conv2d_2/Conv2D/ReadVariableOp%^quant_conv2d_3/Conv2D/ReadVariableOp"^quant_dense/MatMul/ReadVariableOp$^quant_dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::2j
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
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12`
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
$quant_conv2d_2/Conv2D/ReadVariableOp$quant_conv2d_2/Conv2D/ReadVariableOp2L
$quant_conv2d_3/Conv2D/ReadVariableOp$quant_conv2d_3/Conv2D/ReadVariableOp2F
!quant_dense/MatMul/ReadVariableOp!quant_dense/MatMul/ReadVariableOp2J
#quant_dense_1/MatMul/ReadVariableOp#quant_dense_1/MatMul/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€В
 
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
љ0
»
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_201904

inputs
assignmovingavg_201879
assignmovingavg_1_201885)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOp^
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

LogicalAndК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≤
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/201879*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_201879*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp√
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/201879*
_output_shapes
:2
AssignMovingAvg/subЇ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/201879*
_output_shapes
:2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_201879AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/201879*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/201885*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_201885*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЌ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201885*
_output_shapes
:2
AssignMovingAvg_1/subƒ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201885*
_output_shapes
:2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_201885AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/201885*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/add_1≥
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
Ч
§
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_202442

inputs"
matmul_readvariableop_resource
identityИҐMatMul/ReadVariableOpg
ste_sign_10/SignSigninputs*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_10/Signk
ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
ste_sign_10/add/yР
ste_sign_10/addAddV2ste_sign_10/Sign:y:0ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_10/addx
ste_sign_10/Sign_1Signste_sign_10/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_10/Sign_1Г
ste_sign_10/IdentityIdentityste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_10/Identity‘
ste_sign_10/IdentityN	IdentityNste_sign_10/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-202422*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2
ste_sign_10/IdentityNО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpБ
MatMul/ste_sign_9/SignSignMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/Signw
MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
MatMul/ste_sign_9/add/yЯ
MatMul/ste_sign_9/addAddV2MatMul/ste_sign_9/Sign:y:0 MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/addБ
MatMul/ste_sign_9/Sign_1SignMatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/Sign_1М
MatMul/ste_sign_9/IdentityIdentityMatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/Identityл
MatMul/ste_sign_9/IdentityN	IdentityNMatMul/ste_sign_9/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-202432**
_output_shapes
:	А:	А2
MatMul/ste_sign_9/IdentityNТ
MatMulMatMulste_sign_10/IdentityN:output:0$MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€А:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: 
Љ
b
F__inference_activation_layer_call_and_return_conditional_losses_204797

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Љ
ф
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_201216

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constм
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
ћ
І
4__inference_batch_normalization_layer_call_fn_203850

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2009712
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
И
И
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_204612

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOp^
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

LogicalAndУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/add_1№
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
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
О
•
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_201067

inputs"
conv2d_readvariableop_resource
identityИҐConv2D/ReadVariableOp≥
ste_sign_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_2010332
ste_sign_2/PartitionedCallІ
ste_sign_2/IdentityIdentity#ste_sign_2/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
ste_sign_2/IdentityХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOpљ
!Conv2D/ste_sign_1/PartitionedCallPartitionedCallConv2D/ReadVariableOp:value:0*
Tin
2*
Tout
2*&
_output_shapes
:00*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_2010562#
!Conv2D/ste_sign_1/PartitionedCall°
Conv2D/ste_sign_1/IdentityIdentity*Conv2D/ste_sign_1/PartitionedCall:output:0*
T0*&
_output_shapes
:002
Conv2D/ste_sign_1/Identity—
Conv2DConv2Dste_sign_2/Identity:output:0#Conv2D/ste_sign_1/Identity:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*
paddingSAME*
strides
2
Conv2DХ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs:

_output_shapes
: 
–
©
6__inference_batch_normalization_2_layer_call_fn_204284

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2013912
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
€
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_200871

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
—
G
+__inference_ste_sign_1_layer_call_fn_204836

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*&
_output_shapes
:00*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_2010562
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:002

Identity"
identityIdentity:output:0*%
_input_shapes
:00:N J
&
_output_shapes
:00
 
_user_specified_nameinputs
—
G
+__inference_ste_sign_3_layer_call_fn_204870

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*&
_output_shapes
:00*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_2012662
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:002

Identity"
identityIdentity:output:0*%
_input_shapes
:00:N J
&
_output_shapes
:00
 
_user_specified_nameinputs
т
т
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203919

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstЏ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€A
02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€A
0
 
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
Љ
ф
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_204013

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constм
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
б
D
(__inference_flatten_layer_call_fn_204484

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2023412
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€0:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
х%
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204249

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_204234
assignmovingavg_1_204241
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/204234*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/204234*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_204234*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/204234*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/204234*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_204234AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/204234*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/204241*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204241*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_204241*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204241*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204241*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_204241AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/204241*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
Д
І
4__inference_batch_normalization_layer_call_fn_203945

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2020142
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€A
02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€A
0
 
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
Б
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_201081

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
–
©
6__inference_batch_normalization_3_layer_call_fn_204391

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2016362
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
в\
√
F__inference_sequential_layer_call_and_return_conditional_losses_202585
quant_conv2d_input
quant_conv2d_202506
batch_normalization_202510
batch_normalization_202512
batch_normalization_202514
batch_normalization_202516
quant_conv2d_1_202519 
batch_normalization_1_202523 
batch_normalization_1_202525 
batch_normalization_1_202527 
batch_normalization_1_202529
quant_conv2d_2_202532 
batch_normalization_2_202536 
batch_normalization_2_202538 
batch_normalization_2_202540 
batch_normalization_2_202542
quant_conv2d_3_202545 
batch_normalization_3_202549 
batch_normalization_3_202551 
batch_normalization_3_202553 
batch_normalization_3_202555
quant_dense_202559 
batch_normalization_4_202562 
batch_normalization_4_202564 
batch_normalization_4_202566 
batch_normalization_4_202568
quant_dense_1_202571 
batch_normalization_5_202574 
batch_normalization_5_202576 
batch_normalization_5_202578 
batch_normalization_5_202580
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ$quant_conv2d/StatefulPartitionedCallҐ&quant_conv2d_1/StatefulPartitionedCallҐ&quant_conv2d_2/StatefulPartitionedCallҐ&quant_conv2d_3/StatefulPartitionedCallҐ#quant_dense/StatefulPartitionedCallҐ%quant_dense_1/StatefulPartitionedCallв
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputquant_conv2d_202506*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В0*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_2008572&
$quant_conv2d/StatefulPartitionedCall—
max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2008712
max_pooling2d/PartitionedCallи
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_202510batch_normalization_202512batch_normalization_202514batch_normalization_202516*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2020142-
+batch_normalization/StatefulPartitionedCallЛ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0quant_conv2d_1_202519*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2010672(
&quant_conv2d_1/StatefulPartitionedCallў
max_pooling2d_1/PartitionedCallPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2010812!
max_pooling2d_1/PartitionedCallш
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_202523batch_normalization_1_202525batch_normalization_1_202527batch_normalization_1_202529*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2021092/
-batch_normalization_1/StatefulPartitionedCallН
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0quant_conv2d_2_202532*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2012772(
&quant_conv2d_2/StatefulPartitionedCallў
max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2012912!
max_pooling2d_2/PartitionedCallш
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_202536batch_normalization_2_202538batch_normalization_2_202540batch_normalization_2_202542*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2022042/
-batch_normalization_2/StatefulPartitionedCallН
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0quant_conv2d_3_202545*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2014872(
&quant_conv2d_3/StatefulPartitionedCallў
max_pooling2d_3/PartitionedCallPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2015012!
max_pooling2d_3/PartitionedCallш
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_202549batch_normalization_3_202551batch_normalization_3_202553batch_normalization_3_202555*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2022992/
-batch_normalization_3/StatefulPartitionedCallЅ
flatten/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2023412
flatten/PartitionedCallд
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_202559*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_2023722%
#quant_dense/StatefulPartitionedCallх
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_4_202562batch_normalization_4_202564batch_normalization_4_202566batch_normalization_4_202568*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2017882/
-batch_normalization_4/StatefulPartitionedCallБ
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_1_202571*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_2024422'
%quant_dense_1/StatefulPartitionedCallц
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_5_202574batch_normalization_5_202576batch_normalization_5_202578batch_normalization_5_202580*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2019402/
-batch_normalization_5/StatefulPartitionedCall…
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_2024942
activation/PartitionedCallЕ
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall:d `
0
_output_shapes
:€€€€€€€€€В
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
у%
Ъ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203815

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_203800
assignmovingavg_1_203807
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/203800*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/203800*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_203800*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/203800*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/203800*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_203800AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/203800*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/203807*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/203807*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_203807*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/203807*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/203807*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_203807AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/203807*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
О
•
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_201487

inputs"
conv2d_readvariableop_resource
identityИҐConv2D/ReadVariableOp≥
ste_sign_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_6_layer_call_and_return_conditional_losses_2014532
ste_sign_6/PartitionedCallІ
ste_sign_6/IdentityIdentity#ste_sign_6/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
ste_sign_6/IdentityХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOpљ
!Conv2D/ste_sign_5/PartitionedCallPartitionedCallConv2D/ReadVariableOp:value:0*
Tin
2*
Tout
2*&
_output_shapes
:00*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_5_layer_call_and_return_conditional_losses_2014762#
!Conv2D/ste_sign_5/PartitionedCall°
Conv2D/ste_sign_5/IdentityIdentity*Conv2D/ste_sign_5/PartitionedCall:output:0*
T0*&
_output_shapes
:002
Conv2D/ste_sign_5/Identity—
Conv2DConv2Dste_sign_6/Identity:output:0#Conv2D/ste_sign_5/Identity:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*
paddingSAME*
strides
2
Conv2DХ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs:

_output_shapes
: 
М
Ґ
G__inference_quant_dense_layer_call_and_return_conditional_losses_204507

inputs"
matmul_readvariableop_resource
identityИҐMatMul/ReadVariableOpe
ste_sign_8/SignSigninputs*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_8/Signi
ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
ste_sign_8/add/yМ
ste_sign_8/addAddV2ste_sign_8/Sign:y:0ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_8/addu
ste_sign_8/Sign_1Signste_sign_8/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_8/Sign_1А
ste_sign_8/IdentityIdentityste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_8/Identity—
ste_sign_8/IdentityN	IdentityNste_sign_8/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-204487*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2
ste_sign_8/IdentityNП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpВ
MatMul/ste_sign_7/SignSignMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_7/Signw
MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
MatMul/ste_sign_7/add/y†
MatMul/ste_sign_7/addAddV2MatMul/ste_sign_7/Sign:y:0 MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_7/addВ
MatMul/ste_sign_7/Sign_1SignMatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_7/Sign_1Н
MatMul/ste_sign_7/IdentityIdentityMatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_7/Identityн
MatMul/ste_sign_7/IdentityN	IdentityNMatMul/ste_sign_7/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-204497*,
_output_shapes
:
АА:
АА2
MatMul/ste_sign_7/IdentityNТ
MatMulMatMulste_sign_8/IdentityN:output:0$MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€А:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: 
†
r
,__inference_quant_dense_layer_call_fn_204514

inputs
unknown
identityИҐStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_2023722
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€А:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: 
И
©
6__inference_batch_normalization_2_layer_call_fn_204215

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2022042
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€0
 
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
фЯ
≤$
__inference__traced_save_205170
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
@savev2_batch_normalization_2_moving_variance_read_readvariableop4
0savev2_quant_conv2d_3_kernel_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop1
-savev2_quant_dense_kernel_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop3
/savev2_quant_dense_1_kernel_read_readvariableop:
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
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop;
7savev2_adam_quant_conv2d_3_kernel_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop8
4savev2_adam_quant_dense_kernel_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop:
6savev2_adam_quant_dense_1_kernel_m_read_readvariableopA
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
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop;
7savev2_adam_quant_conv2d_3_kernel_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop8
4savev2_adam_quant_dense_kernel_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop:
6savev2_adam_quant_dense_1_kernel_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1•
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b933cb09389349869b4a3c71b44fc057/part2
StringJoin/inputs_1Б

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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename…)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*џ(
value—(Bќ(KB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names°
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ђ
value°BЮKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesъ"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_quant_conv2d_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop0savev2_quant_conv2d_1_kernel_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop0savev2_quant_conv2d_2_kernel_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop0savev2_quant_conv2d_3_kernel_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop-savev2_quant_dense_kernel_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop/savev2_quant_dense_1_kernel_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_quant_conv2d_kernel_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop7savev2_adam_quant_conv2d_1_kernel_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop7savev2_adam_quant_conv2d_2_kernel_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop7savev2_adam_quant_conv2d_3_kernel_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop4savev2_adam_quant_dense_kernel_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop6savev2_adam_quant_dense_1_kernel_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop5savev2_adam_quant_conv2d_kernel_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop7savev2_adam_quant_conv2d_1_kernel_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop7savev2_adam_quant_conv2d_2_kernel_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop7savev2_adam_quant_conv2d_3_kernel_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop4savev2_adam_quant_dense_kernel_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop6savev2_adam_quant_dense_1_kernel_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesѕ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*р
_input_shapesё
џ: :0:0:0:0:0:00:0:0:0:0:00:0:0:0:0:00:0:0:0:0:
АА:А:А:А:А:	А::::: : : : : : : : : :0:0:0:00:0:0:00:0:0:00:0:0:
АА:А:А:	А:::0:0:0:00:0:0:00:0:0:00:0:0:
АА:А:А:	А::: 2(
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
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:00: 

_output_shapes
:0: 

_output_shapes
:0: 	

_output_shapes
:0: 


_output_shapes
:0:,(
&
_output_shapes
:00: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:00: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::
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
:0: )

_output_shapes
:0: *

_output_shapes
:0:,+(
&
_output_shapes
:00: ,

_output_shapes
:0: -

_output_shapes
:0:,.(
&
_output_shapes
:00: /

_output_shapes
:0: 0

_output_shapes
:0:,1(
&
_output_shapes
:00: 2

_output_shapes
:0: 3

_output_shapes
:0:&4"
 
_output_shapes
:
АА:!5

_output_shapes	
:А:!6

_output_shapes	
:А:%7!

_output_shapes
:	А: 8

_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:0: ;

_output_shapes
:0: <

_output_shapes
:0:,=(
&
_output_shapes
:00: >

_output_shapes
:0: ?

_output_shapes
:0:,@(
&
_output_shapes
:00: A

_output_shapes
:0: B

_output_shapes
:0:,C(
&
_output_shapes
:00: D

_output_shapes
:0: E

_output_shapes
:0:&F"
 
_output_shapes
:
АА:!G

_output_shapes	
:А:!H

_output_shapes	
:А:%I!

_output_shapes
:	А: J

_output_shapes
:: K

_output_shapes
::L

_output_shapes
: 
в\
√
F__inference_sequential_layer_call_and_return_conditional_losses_202503
quant_conv2d_input
quant_conv2d_201955
batch_normalization_202041
batch_normalization_202043
batch_normalization_202045
batch_normalization_202047
quant_conv2d_1_202050 
batch_normalization_1_202136 
batch_normalization_1_202138 
batch_normalization_1_202140 
batch_normalization_1_202142
quant_conv2d_2_202145 
batch_normalization_2_202231 
batch_normalization_2_202233 
batch_normalization_2_202235 
batch_normalization_2_202237
quant_conv2d_3_202240 
batch_normalization_3_202326 
batch_normalization_3_202328 
batch_normalization_3_202330 
batch_normalization_3_202332
quant_dense_202381 
batch_normalization_4_202410 
batch_normalization_4_202412 
batch_normalization_4_202414 
batch_normalization_4_202416
quant_dense_1_202451 
batch_normalization_5_202480 
batch_normalization_5_202482 
batch_normalization_5_202484 
batch_normalization_5_202486
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ$quant_conv2d/StatefulPartitionedCallҐ&quant_conv2d_1/StatefulPartitionedCallҐ&quant_conv2d_2/StatefulPartitionedCallҐ&quant_conv2d_3/StatefulPartitionedCallҐ#quant_dense/StatefulPartitionedCallҐ%quant_dense_1/StatefulPartitionedCallв
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputquant_conv2d_201955*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В0*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_2008572&
$quant_conv2d/StatefulPartitionedCall—
max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2008712
max_pooling2d/PartitionedCallи
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_202041batch_normalization_202043batch_normalization_202045batch_normalization_202047*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2019922-
+batch_normalization/StatefulPartitionedCallЛ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0quant_conv2d_1_202050*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2010672(
&quant_conv2d_1/StatefulPartitionedCallў
max_pooling2d_1/PartitionedCallPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2010812!
max_pooling2d_1/PartitionedCallш
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_202136batch_normalization_1_202138batch_normalization_1_202140batch_normalization_1_202142*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2020872/
-batch_normalization_1/StatefulPartitionedCallН
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0quant_conv2d_2_202145*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2012772(
&quant_conv2d_2/StatefulPartitionedCallў
max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2012912!
max_pooling2d_2/PartitionedCallш
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_202231batch_normalization_2_202233batch_normalization_2_202235batch_normalization_2_202237*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2021822/
-batch_normalization_2/StatefulPartitionedCallН
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0quant_conv2d_3_202240*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2014872(
&quant_conv2d_3/StatefulPartitionedCallў
max_pooling2d_3/PartitionedCallPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2015012!
max_pooling2d_3/PartitionedCallш
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_202326batch_normalization_3_202328batch_normalization_3_202330batch_normalization_3_202332*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2022772/
-batch_normalization_3/StatefulPartitionedCallЅ
flatten/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2023412
flatten/PartitionedCallд
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_202381*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_2023722%
#quant_dense/StatefulPartitionedCallх
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_4_202410batch_normalization_4_202412batch_normalization_4_202414batch_normalization_4_202416*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2017522/
-batch_normalization_4/StatefulPartitionedCallБ
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_1_202451*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_2024422'
%quant_dense_1/StatefulPartitionedCallц
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_5_202480batch_normalization_5_202482batch_normalization_5_202484batch_normalization_5_202486*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2019042/
-batch_normalization_5/StatefulPartitionedCall…
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_2024942
activation/PartitionedCallЕ
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall:d `
0
_output_shapes
:€€€€€€€€€В
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
≈	
d
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_201033

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/yy
addAddV2Sign:y:0add/y:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identityв
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-201024*n
_output_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
	IdentityNД

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
х%
Ь
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_203991

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_203976
assignmovingavg_1_203983
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/203976*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/203976*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_203976*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/203976*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/203976*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_203976AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/203976*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/203983*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/203983*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_203983*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/203983*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/203983*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_203983AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/203983*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
Ђ%
Ъ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203897

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_203882
assignmovingavg_1_203889
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/203882*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/203882*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_203882*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/203882*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/203882*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_203882AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/203882*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/203889*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/203889*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_203889*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/203889*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/203889*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_203889AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/203889*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€A
02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€A
0
 
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
И
£
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_200857

inputs"
conv2d_readvariableop_resource
identityИҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOpЈ
Conv2D/ste_sign/PartitionedCallPartitionedCallConv2D/ReadVariableOp:value:0*
Tin
2*
Tout
2*&
_output_shapes
:0*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_ste_sign_layer_call_and_return_conditional_losses_2008462!
Conv2D/ste_sign/PartitionedCallЫ
Conv2D/ste_sign/IdentityIdentity(Conv2D/ste_sign/PartitionedCall:output:0*
T0*&
_output_shapes
:02
Conv2D/ste_sign/Identityє
Conv2DConv2Dinputs!Conv2D/ste_sign/Identity:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*
paddingSAME*
strides
2
Conv2DХ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: 
њ
Ј
+__inference_sequential_layer_call_fn_203704

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
identityИҐStatefulPartitionedCallФ
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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2026702
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€В
 
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
х%
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_201391

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_201376
assignmovingavg_1_201383
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/201376*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/201376*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_201376*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/201376*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/201376*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_201376AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/201376*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/201383*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201383*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_201383*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201383*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201383*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_201383AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/201383*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
≠%
Ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204425

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_204410
assignmovingavg_1_204417
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/204410*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/204410*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_204410*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/204410*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/204410*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_204410AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/204410*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/204417*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204417*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_204417*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204417*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204417*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_204417AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/204417*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€0
 
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
К
u
/__inference_quant_conv2d_3_layer_call_fn_201495

inputs
unknown
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2014872
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs:

_output_shapes
: 
и
©
6__inference_batch_normalization_5_layer_call_fn_204792

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2019402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
ф
ф
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_202109

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstЏ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€ 02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€ 0
 
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
≈	
d
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_204882

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/yy
addAddV2Sign:y:0add/y:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identityв
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-204873*n
_output_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
	IdentityNД

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
ф
ф
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_202299

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstЏ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€0
 
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
м
©
6__inference_batch_normalization_4_layer_call_fn_204638

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2017882
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
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
Ч
§
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_204661

inputs"
matmul_readvariableop_resource
identityИҐMatMul/ReadVariableOpg
ste_sign_10/SignSigninputs*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_10/Signk
ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
ste_sign_10/add/yР
ste_sign_10/addAddV2ste_sign_10/Sign:y:0ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_10/addx
ste_sign_10/Sign_1Signste_sign_10/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_10/Sign_1Г
ste_sign_10/IdentityIdentityste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_10/Identity‘
ste_sign_10/IdentityN	IdentityNste_sign_10/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-204641*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2
ste_sign_10/IdentityNО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpБ
MatMul/ste_sign_9/SignSignMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/Signw
MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
MatMul/ste_sign_9/add/yЯ
MatMul/ste_sign_9/addAddV2MatMul/ste_sign_9/Sign:y:0 MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/addБ
MatMul/ste_sign_9/Sign_1SignMatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/Sign_1М
MatMul/ste_sign_9/IdentityIdentityMatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/Identityл
MatMul/ste_sign_9/IdentityN	IdentityNMatMul/ste_sign_9/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-204651**
_output_shapes
:	А:	А2
MatMul/ste_sign_9/IdentityNТ
MatMulMatMulste_sign_10/IdentityN:output:0$MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€А:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: 
—
d
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_201056

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:002
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:002
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:002
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:002

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-201047*8
_output_shapes&
$:00:002
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:002

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:00:N J
&
_output_shapes
:00
 
_user_specified_nameinputs
Њ
G
+__inference_ste_sign_4_layer_call_fn_204887

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_2012432
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
Љ
ф
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_201636

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constм
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
кк
ј
!__inference__wrapped_model_200828
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
Ksequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource<
8sequential_quant_conv2d_3_conv2d_readvariableop_resource<
8sequential_batch_normalization_3_readvariableop_resource>
:sequential_batch_normalization_3_readvariableop_1_resourceM
Isequential_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource9
5sequential_quant_dense_matmul_readvariableop_resourceF
Bsequential_batch_normalization_4_batchnorm_readvariableop_resourceJ
Fsequential_batch_normalization_4_batchnorm_mul_readvariableop_resourceH
Dsequential_batch_normalization_4_batchnorm_readvariableop_1_resourceH
Dsequential_batch_normalization_4_batchnorm_readvariableop_2_resource;
7sequential_quant_dense_1_matmul_readvariableop_resourceF
Bsequential_batch_normalization_5_batchnorm_readvariableop_resourceJ
Fsequential_batch_normalization_5_batchnorm_mul_readvariableop_resourceH
Dsequential_batch_normalization_5_batchnorm_readvariableop_1_resourceH
Dsequential_batch_normalization_5_batchnorm_readvariableop_2_resource
identityИҐ>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpҐ@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ-sequential/batch_normalization/ReadVariableOpҐ/sequential/batch_normalization/ReadVariableOp_1Ґ@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐBsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ/sequential/batch_normalization_1/ReadVariableOpҐ1sequential/batch_normalization_1/ReadVariableOp_1Ґ@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐBsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ/sequential/batch_normalization_2/ReadVariableOpҐ1sequential/batch_normalization_2/ReadVariableOp_1Ґ@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐBsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ/sequential/batch_normalization_3/ReadVariableOpҐ1sequential/batch_normalization_3/ReadVariableOp_1Ґ9sequential/batch_normalization_4/batchnorm/ReadVariableOpҐ;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1Ґ;sequential/batch_normalization_4/batchnorm/ReadVariableOp_2Ґ=sequential/batch_normalization_4/batchnorm/mul/ReadVariableOpҐ9sequential/batch_normalization_5/batchnorm/ReadVariableOpҐ;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1Ґ;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2Ґ=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOpҐ-sequential/quant_conv2d/Conv2D/ReadVariableOpҐ/sequential/quant_conv2d_1/Conv2D/ReadVariableOpҐ/sequential/quant_conv2d_2/Conv2D/ReadVariableOpҐ/sequential/quant_conv2d_3/Conv2D/ReadVariableOpҐ,sequential/quant_dense/MatMul/ReadVariableOpҐ.sequential/quant_dense_1/MatMul/ReadVariableOpЁ
-sequential/quant_conv2d/Conv2D/ReadVariableOpReadVariableOp6sequential_quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02/
-sequential/quant_conv2d/Conv2D/ReadVariableOpћ
,sequential/quant_conv2d/Conv2D/ste_sign/SignSign5sequential/quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:02.
,sequential/quant_conv2d/Conv2D/ste_sign/Sign£
-sequential/quant_conv2d/Conv2D/ste_sign/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2/
-sequential/quant_conv2d/Conv2D/ste_sign/add/yю
+sequential/quant_conv2d/Conv2D/ste_sign/addAddV20sequential/quant_conv2d/Conv2D/ste_sign/Sign:y:06sequential/quant_conv2d/Conv2D/ste_sign/add/y:output:0*
T0*&
_output_shapes
:02-
+sequential/quant_conv2d/Conv2D/ste_sign/add 
.sequential/quant_conv2d/Conv2D/ste_sign/Sign_1Sign/sequential/quant_conv2d/Conv2D/ste_sign/add:z:0*
T0*&
_output_shapes
:020
.sequential/quant_conv2d/Conv2D/ste_sign/Sign_1’
0sequential/quant_conv2d/Conv2D/ste_sign/IdentityIdentity2sequential/quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*
T0*&
_output_shapes
:022
0sequential/quant_conv2d/Conv2D/ste_sign/Identity”
1sequential/quant_conv2d/Conv2D/ste_sign/IdentityN	IdentityN2sequential/quant_conv2d/Conv2D/ste_sign/Sign_1:y:05sequential/quant_conv2d/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-200606*8
_output_shapes&
$:0:023
1sequential/quant_conv2d/Conv2D/ste_sign/IdentityNэ
sequential/quant_conv2d/Conv2DConv2Dquant_conv2d_input:sequential/quant_conv2d/Conv2D/ste_sign/IdentityN:output:0*
T0*0
_output_shapes
:€€€€€€€€€В0*
paddingSAME*
strides
2 
sequential/quant_conv2d/Conv2Dе
 sequential/max_pooling2d/MaxPoolMaxPool'sequential/quant_conv2d/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€A
0*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolЬ
+sequential/batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2-
+sequential/batch_normalization/LogicalAnd/xЬ
+sequential/batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2-
+sequential/batch_normalization/LogicalAnd/yи
)sequential/batch_normalization/LogicalAnd
LogicalAnd4sequential/batch_normalization/LogicalAnd/x:output:04sequential/batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2+
)sequential/batch_normalization/LogicalAnd—
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:0*
dtype02/
-sequential/batch_normalization/ReadVariableOp„
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:0*
dtype021
/sequential/batch_normalization/ReadVariableOp_1Д
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpК
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1І
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3)sequential/max_pooling2d/MaxPool:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
0:0:0:0:0:*
epsilon%oГ:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3С
$sequential/batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2&
$sequential/batch_normalization/ConstЌ
)sequential/quant_conv2d_1/ste_sign_2/SignSign3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€A
02+
)sequential/quant_conv2d_1/ste_sign_2/SignЭ
*sequential/quant_conv2d_1/ste_sign_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2,
*sequential/quant_conv2d_1/ste_sign_2/add/yы
(sequential/quant_conv2d_1/ste_sign_2/addAddV2-sequential/quant_conv2d_1/ste_sign_2/Sign:y:03sequential/quant_conv2d_1/ste_sign_2/add/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
02*
(sequential/quant_conv2d_1/ste_sign_2/add 
+sequential/quant_conv2d_1/ste_sign_2/Sign_1Sign,sequential/quant_conv2d_1/ste_sign_2/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€A
02-
+sequential/quant_conv2d_1/ste_sign_2/Sign_1’
-sequential/quant_conv2d_1/ste_sign_2/IdentityIdentity/sequential/quant_conv2d_1/ste_sign_2/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€A
02/
-sequential/quant_conv2d_1/ste_sign_2/IdentityЏ
.sequential/quant_conv2d_1/ste_sign_2/IdentityN	IdentityN/sequential/quant_conv2d_1/ste_sign_2/Sign_1:y:03sequential/batch_normalization/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-200634*J
_output_shapes8
6:€€€€€€€€€A
0:€€€€€€€€€A
020
.sequential/quant_conv2d_1/ste_sign_2/IdentityNг
/sequential/quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp8sequential_quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/sequential/quant_conv2d_1/Conv2D/ReadVariableOp÷
0sequential/quant_conv2d_1/Conv2D/ste_sign_1/SignSign7sequential/quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0022
0sequential/quant_conv2d_1/Conv2D/ste_sign_1/SignЂ
1sequential/quant_conv2d_1/Conv2D/ste_sign_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=23
1sequential/quant_conv2d_1/Conv2D/ste_sign_1/add/yО
/sequential/quant_conv2d_1/Conv2D/ste_sign_1/addAddV24sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign:y:0:sequential/quant_conv2d_1/Conv2D/ste_sign_1/add/y:output:0*
T0*&
_output_shapes
:0021
/sequential/quant_conv2d_1/Conv2D/ste_sign_1/add÷
2sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1Sign3sequential/quant_conv2d_1/Conv2D/ste_sign_1/add:z:0*
T0*&
_output_shapes
:0024
2sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1б
4sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityIdentity6sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0*
T0*&
_output_shapes
:0026
4sequential/quant_conv2d_1/Conv2D/ste_sign_1/Identityб
5sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityN	IdentityN6sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:07sequential/quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-200644*8
_output_shapes&
$:00:0027
5sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityN©
 sequential/quant_conv2d_1/Conv2DConv2D7sequential/quant_conv2d_1/ste_sign_2/IdentityN:output:0>sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
0*
paddingSAME*
strides
2"
 sequential/quant_conv2d_1/Conv2Dл
"sequential/max_pooling2d_1/MaxPoolMaxPool)sequential/quant_conv2d_1/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€ 0*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool†
-sequential/batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-sequential/batch_normalization_1/LogicalAnd/x†
-sequential/batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-sequential/batch_normalization_1/LogicalAnd/yр
+sequential/batch_normalization_1/LogicalAnd
LogicalAnd6sequential/batch_normalization_1/LogicalAnd/x:output:06sequential/batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_1/LogicalAnd„
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype021
/sequential/batch_normalization_1/ReadVariableOpЁ
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1К
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpР
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1µ
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3+sequential/max_pooling2d_1/MaxPool:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 0:0:0:0:0:*
epsilon%oГ:*
is_training( 23
1sequential/batch_normalization_1/FusedBatchNormV3Х
&sequential/batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2(
&sequential/batch_normalization_1/Constѕ
)sequential/quant_conv2d_2/ste_sign_4/SignSign5sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 02+
)sequential/quant_conv2d_2/ste_sign_4/SignЭ
*sequential/quant_conv2d_2/ste_sign_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2,
*sequential/quant_conv2d_2/ste_sign_4/add/yы
(sequential/quant_conv2d_2/ste_sign_4/addAddV2-sequential/quant_conv2d_2/ste_sign_4/Sign:y:03sequential/quant_conv2d_2/ste_sign_4/add/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 02*
(sequential/quant_conv2d_2/ste_sign_4/add 
+sequential/quant_conv2d_2/ste_sign_4/Sign_1Sign,sequential/quant_conv2d_2/ste_sign_4/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€ 02-
+sequential/quant_conv2d_2/ste_sign_4/Sign_1’
-sequential/quant_conv2d_2/ste_sign_4/IdentityIdentity/sequential/quant_conv2d_2/ste_sign_4/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 02/
-sequential/quant_conv2d_2/ste_sign_4/Identity№
.sequential/quant_conv2d_2/ste_sign_4/IdentityN	IdentityN/sequential/quant_conv2d_2/ste_sign_4/Sign_1:y:05sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-200672*J
_output_shapes8
6:€€€€€€€€€ 0:€€€€€€€€€ 020
.sequential/quant_conv2d_2/ste_sign_4/IdentityNг
/sequential/quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp8sequential_quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/sequential/quant_conv2d_2/Conv2D/ReadVariableOp÷
0sequential/quant_conv2d_2/Conv2D/ste_sign_3/SignSign7sequential/quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0022
0sequential/quant_conv2d_2/Conv2D/ste_sign_3/SignЂ
1sequential/quant_conv2d_2/Conv2D/ste_sign_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=23
1sequential/quant_conv2d_2/Conv2D/ste_sign_3/add/yО
/sequential/quant_conv2d_2/Conv2D/ste_sign_3/addAddV24sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign:y:0:sequential/quant_conv2d_2/Conv2D/ste_sign_3/add/y:output:0*
T0*&
_output_shapes
:0021
/sequential/quant_conv2d_2/Conv2D/ste_sign_3/add÷
2sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1Sign3sequential/quant_conv2d_2/Conv2D/ste_sign_3/add:z:0*
T0*&
_output_shapes
:0024
2sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1б
4sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityIdentity6sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0*
T0*&
_output_shapes
:0026
4sequential/quant_conv2d_2/Conv2D/ste_sign_3/Identityб
5sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityN	IdentityN6sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:07sequential/quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-200682*8
_output_shapes&
$:00:0027
5sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityN©
 sequential/quant_conv2d_2/Conv2DConv2D7sequential/quant_conv2d_2/ste_sign_4/IdentityN:output:0>sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 0*
paddingSAME*
strides
2"
 sequential/quant_conv2d_2/Conv2Dл
"sequential/max_pooling2d_2/MaxPoolMaxPool)sequential/quant_conv2d_2/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€0*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPool†
-sequential/batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-sequential/batch_normalization_2/LogicalAnd/x†
-sequential/batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-sequential/batch_normalization_2/LogicalAnd/yр
+sequential/batch_normalization_2/LogicalAnd
LogicalAnd6sequential/batch_normalization_2/LogicalAnd/x:output:06sequential/batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_2/LogicalAnd„
/sequential/batch_normalization_2/ReadVariableOpReadVariableOp8sequential_batch_normalization_2_readvariableop_resource*
_output_shapes
:0*
dtype021
/sequential/batch_normalization_2/ReadVariableOpЁ
1sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:0*
dtype023
1sequential/batch_normalization_2/ReadVariableOp_1К
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02B
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpР
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02D
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1µ
1sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3+sequential/max_pooling2d_2/MaxPool:output:07sequential/batch_normalization_2/ReadVariableOp:value:09sequential/batch_normalization_2/ReadVariableOp_1:value:0Hsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 23
1sequential/batch_normalization_2/FusedBatchNormV3Х
&sequential/batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2(
&sequential/batch_normalization_2/Constѕ
)sequential/quant_conv2d_3/ste_sign_6/SignSign5sequential/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€02+
)sequential/quant_conv2d_3/ste_sign_6/SignЭ
*sequential/quant_conv2d_3/ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2,
*sequential/quant_conv2d_3/ste_sign_6/add/yы
(sequential/quant_conv2d_3/ste_sign_6/addAddV2-sequential/quant_conv2d_3/ste_sign_6/Sign:y:03sequential/quant_conv2d_3/ste_sign_6/add/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€02*
(sequential/quant_conv2d_3/ste_sign_6/add 
+sequential/quant_conv2d_3/ste_sign_6/Sign_1Sign,sequential/quant_conv2d_3/ste_sign_6/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€02-
+sequential/quant_conv2d_3/ste_sign_6/Sign_1’
-sequential/quant_conv2d_3/ste_sign_6/IdentityIdentity/sequential/quant_conv2d_3/ste_sign_6/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€02/
-sequential/quant_conv2d_3/ste_sign_6/Identity№
.sequential/quant_conv2d_3/ste_sign_6/IdentityN	IdentityN/sequential/quant_conv2d_3/ste_sign_6/Sign_1:y:05sequential/batch_normalization_2/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-200710*J
_output_shapes8
6:€€€€€€€€€0:€€€€€€€€€020
.sequential/quant_conv2d_3/ste_sign_6/IdentityNг
/sequential/quant_conv2d_3/Conv2D/ReadVariableOpReadVariableOp8sequential_quant_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype021
/sequential/quant_conv2d_3/Conv2D/ReadVariableOp÷
0sequential/quant_conv2d_3/Conv2D/ste_sign_5/SignSign7sequential/quant_conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:0022
0sequential/quant_conv2d_3/Conv2D/ste_sign_5/SignЂ
1sequential/quant_conv2d_3/Conv2D/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=23
1sequential/quant_conv2d_3/Conv2D/ste_sign_5/add/yО
/sequential/quant_conv2d_3/Conv2D/ste_sign_5/addAddV24sequential/quant_conv2d_3/Conv2D/ste_sign_5/Sign:y:0:sequential/quant_conv2d_3/Conv2D/ste_sign_5/add/y:output:0*
T0*&
_output_shapes
:0021
/sequential/quant_conv2d_3/Conv2D/ste_sign_5/add÷
2sequential/quant_conv2d_3/Conv2D/ste_sign_5/Sign_1Sign3sequential/quant_conv2d_3/Conv2D/ste_sign_5/add:z:0*
T0*&
_output_shapes
:0024
2sequential/quant_conv2d_3/Conv2D/ste_sign_5/Sign_1б
4sequential/quant_conv2d_3/Conv2D/ste_sign_5/IdentityIdentity6sequential/quant_conv2d_3/Conv2D/ste_sign_5/Sign_1:y:0*
T0*&
_output_shapes
:0026
4sequential/quant_conv2d_3/Conv2D/ste_sign_5/Identityб
5sequential/quant_conv2d_3/Conv2D/ste_sign_5/IdentityN	IdentityN6sequential/quant_conv2d_3/Conv2D/ste_sign_5/Sign_1:y:07sequential/quant_conv2d_3/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-200720*8
_output_shapes&
$:00:0027
5sequential/quant_conv2d_3/Conv2D/ste_sign_5/IdentityN©
 sequential/quant_conv2d_3/Conv2DConv2D7sequential/quant_conv2d_3/ste_sign_6/IdentityN:output:0>sequential/quant_conv2d_3/Conv2D/ste_sign_5/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€0*
paddingSAME*
strides
2"
 sequential/quant_conv2d_3/Conv2Dл
"sequential/max_pooling2d_3/MaxPoolMaxPool)sequential/quant_conv2d_3/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€0*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_3/MaxPool†
-sequential/batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-sequential/batch_normalization_3/LogicalAnd/x†
-sequential/batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-sequential/batch_normalization_3/LogicalAnd/yр
+sequential/batch_normalization_3/LogicalAnd
LogicalAnd6sequential/batch_normalization_3/LogicalAnd/x:output:06sequential/batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_3/LogicalAnd„
/sequential/batch_normalization_3/ReadVariableOpReadVariableOp8sequential_batch_normalization_3_readvariableop_resource*
_output_shapes
:0*
dtype021
/sequential/batch_normalization_3/ReadVariableOpЁ
1sequential/batch_normalization_3/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:0*
dtype023
1sequential/batch_normalization_3/ReadVariableOp_1К
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02B
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpР
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02D
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1µ
1sequential/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3+sequential/max_pooling2d_3/MaxPool:output:07sequential/batch_normalization_3/ReadVariableOp:value:09sequential/batch_normalization_3/ReadVariableOp_1:value:0Hsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 23
1sequential/batch_normalization_3/FusedBatchNormV3Х
&sequential/batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2(
&sequential/batch_normalization_3/ConstЕ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
sequential/flatten/Const–
sequential/flatten/ReshapeReshape5sequential/batch_normalization_3/FusedBatchNormV3:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/flatten/Reshape∞
&sequential/quant_dense/ste_sign_8/SignSign#sequential/flatten/Reshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential/quant_dense/ste_sign_8/SignЧ
'sequential/quant_dense/ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2)
'sequential/quant_dense/ste_sign_8/add/yи
%sequential/quant_dense/ste_sign_8/addAddV2*sequential/quant_dense/ste_sign_8/Sign:y:00sequential/quant_dense/ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%sequential/quant_dense/ste_sign_8/addЇ
(sequential/quant_dense/ste_sign_8/Sign_1Sign)sequential/quant_dense/ste_sign_8/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(sequential/quant_dense/ste_sign_8/Sign_1≈
*sequential/quant_dense/ste_sign_8/IdentityIdentity,sequential/quant_dense/ste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2,
*sequential/quant_dense/ste_sign_8/Identity≥
+sequential/quant_dense/ste_sign_8/IdentityN	IdentityN,sequential/quant_dense/ste_sign_8/Sign_1:y:0#sequential/flatten/Reshape:output:0*
T
2*,
_gradient_op_typeCustomGradient-200750*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2-
+sequential/quant_dense/ste_sign_8/IdentityN‘
,sequential/quant_dense/MatMul/ReadVariableOpReadVariableOp5sequential_quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential/quant_dense/MatMul/ReadVariableOp«
-sequential/quant_dense/MatMul/ste_sign_7/SignSign4sequential/quant_dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2/
-sequential/quant_dense/MatMul/ste_sign_7/Sign•
.sequential/quant_dense/MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=20
.sequential/quant_dense/MatMul/ste_sign_7/add/yь
,sequential/quant_dense/MatMul/ste_sign_7/addAddV21sequential/quant_dense/MatMul/ste_sign_7/Sign:y:07sequential/quant_dense/MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
АА2.
,sequential/quant_dense/MatMul/ste_sign_7/add«
/sequential/quant_dense/MatMul/ste_sign_7/Sign_1Sign0sequential/quant_dense/MatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
АА21
/sequential/quant_dense/MatMul/ste_sign_7/Sign_1“
1sequential/quant_dense/MatMul/ste_sign_7/IdentityIdentity3sequential/quant_dense/MatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
АА23
1sequential/quant_dense/MatMul/ste_sign_7/Identity…
2sequential/quant_dense/MatMul/ste_sign_7/IdentityN	IdentityN3sequential/quant_dense/MatMul/ste_sign_7/Sign_1:y:04sequential/quant_dense/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-200760*,
_output_shapes
:
АА:
АА24
2sequential/quant_dense/MatMul/ste_sign_7/IdentityNо
sequential/quant_dense/MatMulMatMul4sequential/quant_dense/ste_sign_8/IdentityN:output:0;sequential/quant_dense/MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/quant_dense/MatMul†
-sequential/batch_normalization_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-sequential/batch_normalization_4/LogicalAnd/x†
-sequential/batch_normalization_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-sequential/batch_normalization_4/LogicalAnd/yр
+sequential/batch_normalization_4/LogicalAnd
LogicalAnd6sequential/batch_normalization_4/LogicalAnd/x:output:06sequential/batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_4/LogicalAndц
9sequential/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpBsequential_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9sequential/batch_normalization_4/batchnorm/ReadVariableOp©
0sequential/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:22
0sequential/batch_normalization_4/batchnorm/add/yН
.sequential/batch_normalization_4/batchnorm/addAddV2Asequential/batch_normalization_4/batchnorm/ReadVariableOp:value:09sequential/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А20
.sequential/batch_normalization_4/batchnorm/add«
0sequential/batch_normalization_4/batchnorm/RsqrtRsqrt2sequential/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А22
0sequential/batch_normalization_4/batchnorm/RsqrtВ
=sequential/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpFsequential_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02?
=sequential/batch_normalization_4/batchnorm/mul/ReadVariableOpК
.sequential/batch_normalization_4/batchnorm/mulMul4sequential/batch_normalization_4/batchnorm/Rsqrt:y:0Esequential/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А20
.sequential/batch_normalization_4/batchnorm/mulы
0sequential/batch_normalization_4/batchnorm/mul_1Mul'sequential/quant_dense/MatMul:product:02sequential/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А22
0sequential/batch_normalization_4/batchnorm/mul_1ь
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02=
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1К
0sequential/batch_normalization_4/batchnorm/mul_2MulCsequential/batch_normalization_4/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А22
0sequential/batch_normalization_4/batchnorm/mul_2ь
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpDsequential_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02=
;sequential/batch_normalization_4/batchnorm/ReadVariableOp_2И
.sequential/batch_normalization_4/batchnorm/subSubCsequential/batch_normalization_4/batchnorm/ReadVariableOp_2:value:04sequential/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А20
.sequential/batch_normalization_4/batchnorm/subК
0sequential/batch_normalization_4/batchnorm/add_1AddV24sequential/batch_normalization_4/batchnorm/mul_1:z:02sequential/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А22
0sequential/batch_normalization_4/batchnorm/add_1«
)sequential/quant_dense_1/ste_sign_10/SignSign4sequential/batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2+
)sequential/quant_dense_1/ste_sign_10/SignЭ
*sequential/quant_dense_1/ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2,
*sequential/quant_dense_1/ste_sign_10/add/yф
(sequential/quant_dense_1/ste_sign_10/addAddV2-sequential/quant_dense_1/ste_sign_10/Sign:y:03sequential/quant_dense_1/ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(sequential/quant_dense_1/ste_sign_10/add√
+sequential/quant_dense_1/ste_sign_10/Sign_1Sign,sequential/quant_dense_1/ste_sign_10/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2-
+sequential/quant_dense_1/ste_sign_10/Sign_1ќ
-sequential/quant_dense_1/ste_sign_10/IdentityIdentity/sequential/quant_dense_1/ste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2/
-sequential/quant_dense_1/ste_sign_10/IdentityЌ
.sequential/quant_dense_1/ste_sign_10/IdentityN	IdentityN/sequential/quant_dense_1/ste_sign_10/Sign_1:y:04sequential/batch_normalization_4/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-200788*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А20
.sequential/quant_dense_1/ste_sign_10/IdentityNў
.sequential/quant_dense_1/MatMul/ReadVariableOpReadVariableOp7sequential_quant_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype020
.sequential/quant_dense_1/MatMul/ReadVariableOpћ
/sequential/quant_dense_1/MatMul/ste_sign_9/SignSign6sequential/quant_dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А21
/sequential/quant_dense_1/MatMul/ste_sign_9/Sign©
0sequential/quant_dense_1/MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=22
0sequential/quant_dense_1/MatMul/ste_sign_9/add/yГ
.sequential/quant_dense_1/MatMul/ste_sign_9/addAddV23sequential/quant_dense_1/MatMul/ste_sign_9/Sign:y:09sequential/quant_dense_1/MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	А20
.sequential/quant_dense_1/MatMul/ste_sign_9/addћ
1sequential/quant_dense_1/MatMul/ste_sign_9/Sign_1Sign2sequential/quant_dense_1/MatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	А23
1sequential/quant_dense_1/MatMul/ste_sign_9/Sign_1„
3sequential/quant_dense_1/MatMul/ste_sign_9/IdentityIdentity5sequential/quant_dense_1/MatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	А25
3sequential/quant_dense_1/MatMul/ste_sign_9/Identityѕ
4sequential/quant_dense_1/MatMul/ste_sign_9/IdentityN	IdentityN5sequential/quant_dense_1/MatMul/ste_sign_9/Sign_1:y:06sequential/quant_dense_1/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-200798**
_output_shapes
:	А:	А26
4sequential/quant_dense_1/MatMul/ste_sign_9/IdentityNц
sequential/quant_dense_1/MatMulMatMul7sequential/quant_dense_1/ste_sign_10/IdentityN:output:0=sequential/quant_dense_1/MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€2!
sequential/quant_dense_1/MatMul†
-sequential/batch_normalization_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-sequential/batch_normalization_5/LogicalAnd/x†
-sequential/batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-sequential/batch_normalization_5/LogicalAnd/yр
+sequential/batch_normalization_5/LogicalAnd
LogicalAnd6sequential/batch_normalization_5/LogicalAnd/x:output:06sequential/batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2-
+sequential/batch_normalization_5/LogicalAndх
9sequential/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOpBsequential_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02;
9sequential/batch_normalization_5/batchnorm/ReadVariableOp©
0sequential/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:22
0sequential/batch_normalization_5/batchnorm/add/yМ
.sequential/batch_normalization_5/batchnorm/addAddV2Asequential/batch_normalization_5/batchnorm/ReadVariableOp:value:09sequential/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:20
.sequential/batch_normalization_5/batchnorm/add∆
0sequential/batch_normalization_5/batchnorm/RsqrtRsqrt2sequential/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:22
0sequential/batch_normalization_5/batchnorm/RsqrtБ
=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpFsequential_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOpЙ
.sequential/batch_normalization_5/batchnorm/mulMul4sequential/batch_normalization_5/batchnorm/Rsqrt:y:0Esequential/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.sequential/batch_normalization_5/batchnorm/mulь
0sequential/batch_normalization_5/batchnorm/mul_1Mul)sequential/quant_dense_1/MatMul:product:02sequential/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
0sequential/batch_normalization_5/batchnorm/mul_1ы
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1Й
0sequential/batch_normalization_5/batchnorm/mul_2MulCsequential/batch_normalization_5/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:22
0sequential/batch_normalization_5/batchnorm/mul_2ы
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpDsequential_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02=
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2З
.sequential/batch_normalization_5/batchnorm/subSubCsequential/batch_normalization_5/batchnorm/ReadVariableOp_2:value:04sequential/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization_5/batchnorm/subЙ
0sequential/batch_normalization_5/batchnorm/add_1AddV24sequential/batch_normalization_5/batchnorm/mul_1:z:02sequential/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
0sequential/batch_normalization_5/batchnorm/add_1±
sequential/activation/SoftmaxSoftmax4sequential/batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential/activation/SoftmaxЅ
IdentityIdentity'sequential/activation/Softmax:softmax:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1A^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_2/ReadVariableOp2^sequential/batch_normalization_2/ReadVariableOp_1A^sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_3/ReadVariableOp2^sequential/batch_normalization_3/ReadVariableOp_1:^sequential/batch_normalization_4/batchnorm/ReadVariableOp<^sequential/batch_normalization_4/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_4/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_4/batchnorm/mul/ReadVariableOp:^sequential/batch_normalization_5/batchnorm/ReadVariableOp<^sequential/batch_normalization_5/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_5/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_5/batchnorm/mul/ReadVariableOp.^sequential/quant_conv2d/Conv2D/ReadVariableOp0^sequential/quant_conv2d_1/Conv2D/ReadVariableOp0^sequential/quant_conv2d_2/Conv2D/ReadVariableOp0^sequential/quant_conv2d_3/Conv2D/ReadVariableOp-^sequential/quant_dense/MatMul/ReadVariableOp/^sequential/quant_dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::2А
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2Д
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12Д
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2И
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12Д
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2И
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_2/ReadVariableOp/sequential/batch_normalization_2/ReadVariableOp2f
1sequential/batch_normalization_2/ReadVariableOp_11sequential/batch_normalization_2/ReadVariableOp_12Д
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2И
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_3/ReadVariableOp/sequential/batch_normalization_3/ReadVariableOp2f
1sequential/batch_normalization_3/ReadVariableOp_11sequential/batch_normalization_3/ReadVariableOp_12v
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
/sequential/quant_conv2d_2/Conv2D/ReadVariableOp/sequential/quant_conv2d_2/Conv2D/ReadVariableOp2b
/sequential/quant_conv2d_3/Conv2D/ReadVariableOp/sequential/quant_conv2d_3/Conv2D/ReadVariableOp2\
,sequential/quant_dense/MatMul/ReadVariableOp,sequential/quant_dense/MatMul/ReadVariableOp2`
.sequential/quant_dense_1/MatMul/ReadVariableOp.sequential/quant_dense_1/MatMul/ReadVariableOp:d `
0
_output_shapes
:€€€€€€€€€В
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
—
d
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_204865

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:002
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:002
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:002
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:002

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-204856*8
_output_shapes&
$:00:002
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:002

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:00:N J
&
_output_shapes
:00
 
_user_specified_nameinputs
–
©
6__inference_batch_normalization_1_layer_call_fn_204039

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2012162
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
Б
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_201291

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ип
й
F__inference_sequential_layer_call_and_return_conditional_losses_203412

inputs/
+quant_conv2d_conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource.
*batch_normalization_assignmovingavg_2031360
,batch_normalization_assignmovingavg_1_2031431
-quant_conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resource0
,batch_normalization_1_assignmovingavg_2031862
.batch_normalization_1_assignmovingavg_1_2031931
-quant_conv2d_2_conv2d_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resource0
,batch_normalization_2_assignmovingavg_2032362
.batch_normalization_2_assignmovingavg_1_2032431
-quant_conv2d_3_conv2d_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resource0
,batch_normalization_3_assignmovingavg_2032862
.batch_normalization_3_assignmovingavg_1_203293.
*quant_dense_matmul_readvariableop_resource0
,batch_normalization_4_assignmovingavg_2033322
.batch_normalization_4_assignmovingavg_1_203338?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource0
,quant_dense_1_matmul_readvariableop_resource0
,batch_normalization_5_assignmovingavg_2033862
.batch_normalization_5_assignmovingavg_1_203392?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource
identityИҐ7batch_normalization/AssignMovingAvg/AssignSubVariableOpҐ2batch_normalization/AssignMovingAvg/ReadVariableOpҐ9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpҐ4batch_normalization/AssignMovingAvg_1/ReadVariableOpҐ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґ9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ$batch_normalization_3/ReadVariableOpҐ&batch_normalization_3/ReadVariableOp_1Ґ9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_4/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_4/batchnorm/ReadVariableOpҐ2batch_normalization_4/batchnorm/mul/ReadVariableOpҐ9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_5/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_5/batchnorm/ReadVariableOpҐ2batch_normalization_5/batchnorm/mul/ReadVariableOpҐ"quant_conv2d/Conv2D/ReadVariableOpҐ$quant_conv2d_1/Conv2D/ReadVariableOpҐ$quant_conv2d_2/Conv2D/ReadVariableOpҐ$quant_conv2d_3/Conv2D/ReadVariableOpҐ!quant_dense/MatMul/ReadVariableOpҐ#quant_dense_1/MatMul/ReadVariableOpЉ
"quant_conv2d/Conv2D/ReadVariableOpReadVariableOp+quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02$
"quant_conv2d/Conv2D/ReadVariableOpЂ
!quant_conv2d/Conv2D/ste_sign/SignSign*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:02#
!quant_conv2d/Conv2D/ste_sign/SignН
"quant_conv2d/Conv2D/ste_sign/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2$
"quant_conv2d/Conv2D/ste_sign/add/y“
 quant_conv2d/Conv2D/ste_sign/addAddV2%quant_conv2d/Conv2D/ste_sign/Sign:y:0+quant_conv2d/Conv2D/ste_sign/add/y:output:0*
T0*&
_output_shapes
:02"
 quant_conv2d/Conv2D/ste_sign/add©
#quant_conv2d/Conv2D/ste_sign/Sign_1Sign$quant_conv2d/Conv2D/ste_sign/add:z:0*
T0*&
_output_shapes
:02%
#quant_conv2d/Conv2D/ste_sign/Sign_1і
%quant_conv2d/Conv2D/ste_sign/IdentityIdentity'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*
T0*&
_output_shapes
:02'
%quant_conv2d/Conv2D/ste_sign/IdentityІ
&quant_conv2d/Conv2D/ste_sign/IdentityN	IdentityN'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203110*8
_output_shapes&
$:0:02(
&quant_conv2d/Conv2D/ste_sign/IdentityN–
quant_conv2d/Conv2DConv2Dinputs/quant_conv2d/Conv2D/ste_sign/IdentityN:output:0*
T0*0
_output_shapes
:€€€€€€€€€В0*
paddingSAME*
strides
2
quant_conv2d/Conv2Dƒ
max_pooling2d/MaxPoolMaxPoolquant_conv2d/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€A
0*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolЖ
 batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 batch_normalization/LogicalAnd/xЖ
 batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 batch_normalization/LogicalAnd/yЉ
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2 
batch_normalization/LogicalAnd∞
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:0*
dtype02$
"batch_normalization/ReadVariableOpґ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:0*
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
batch_normalization/Const_1Х
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0"batch_normalization/Const:output:0$batch_normalization/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
0:0:0:0:0:*
epsilon%oГ:2&
$batch_normalization/FusedBatchNormV3
batch_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization/Const_2Џ
)batch_normalization/AssignMovingAvg/sub/xConst*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/203136*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)batch_normalization/AssignMovingAvg/sub/xУ
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/203136*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/subѕ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_assignmovingavg_203136*
_output_shapes
:0*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp∞
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/203136*
_output_shapes
:02+
)batch_normalization/AssignMovingAvg/sub_1Щ
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/203136*
_output_shapes
:02)
'batch_normalization/AssignMovingAvg/mulщ
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_assignmovingavg_203136+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/203136*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpа
+batch_normalization/AssignMovingAvg_1/sub/xConst*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/203143*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization/AssignMovingAvg_1/sub/xЫ
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/203143*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/sub’
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_assignmovingavg_1_203143*
_output_shapes
:0*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpЉ
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/203143*
_output_shapes
:02-
+batch_normalization/AssignMovingAvg_1/sub_1£
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/203143*
_output_shapes
:02+
)batch_normalization/AssignMovingAvg_1/mulЕ
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_assignmovingavg_1_203143-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/203143*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpђ
quant_conv2d_1/ste_sign_2/SignSign(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€A
02 
quant_conv2d_1/ste_sign_2/SignЗ
quant_conv2d_1/ste_sign_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2!
quant_conv2d_1/ste_sign_2/add/yѕ
quant_conv2d_1/ste_sign_2/addAddV2"quant_conv2d_1/ste_sign_2/Sign:y:0(quant_conv2d_1/ste_sign_2/add/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
02
quant_conv2d_1/ste_sign_2/add©
 quant_conv2d_1/ste_sign_2/Sign_1Sign!quant_conv2d_1/ste_sign_2/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€A
02"
 quant_conv2d_1/ste_sign_2/Sign_1і
"quant_conv2d_1/ste_sign_2/IdentityIdentity$quant_conv2d_1/ste_sign_2/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€A
02$
"quant_conv2d_1/ste_sign_2/IdentityЃ
#quant_conv2d_1/ste_sign_2/IdentityN	IdentityN$quant_conv2d_1/ste_sign_2/Sign_1:y:0(batch_normalization/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-203150*J
_output_shapes8
6:€€€€€€€€€A
0:€€€€€€€€€A
02%
#quant_conv2d_1/ste_sign_2/IdentityN¬
$quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02&
$quant_conv2d_1/Conv2D/ReadVariableOpµ
%quant_conv2d_1/Conv2D/ste_sign_1/SignSign,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:002'
%quant_conv2d_1/Conv2D/ste_sign_1/SignХ
&quant_conv2d_1/Conv2D/ste_sign_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2(
&quant_conv2d_1/Conv2D/ste_sign_1/add/yв
$quant_conv2d_1/Conv2D/ste_sign_1/addAddV2)quant_conv2d_1/Conv2D/ste_sign_1/Sign:y:0/quant_conv2d_1/Conv2D/ste_sign_1/add/y:output:0*
T0*&
_output_shapes
:002&
$quant_conv2d_1/Conv2D/ste_sign_1/addµ
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1Sign(quant_conv2d_1/Conv2D/ste_sign_1/add:z:0*
T0*&
_output_shapes
:002)
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1ј
)quant_conv2d_1/Conv2D/ste_sign_1/IdentityIdentity+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0*
T0*&
_output_shapes
:002+
)quant_conv2d_1/Conv2D/ste_sign_1/Identityµ
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityN	IdentityN+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203160*8
_output_shapes&
$:00:002,
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityNэ
quant_conv2d_1/Conv2DConv2D,quant_conv2d_1/ste_sign_2/IdentityN:output:03quant_conv2d_1/Conv2D/ste_sign_1/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
0*
paddingSAME*
strides
2
quant_conv2d_1/Conv2D 
max_pooling2d_1/MaxPoolMaxPoolquant_conv2d_1/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€ 0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolК
"batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_1/LogicalAnd/xК
"batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_1/LogicalAnd/yƒ
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_1/LogicalAndґ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_1/ReadVariableOpЉ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_1/ReadVariableOp_1}
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_1/ConstБ
batch_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_1/Const_1£
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0$batch_normalization_1/Const:output:0&batch_normalization_1/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 0:0:0:0:0:*
epsilon%oГ:2(
&batch_normalization_1/FusedBatchNormV3Г
batch_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_1/Const_2а
+batch_normalization_1/AssignMovingAvg/sub/xConst*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/203186*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_1/AssignMovingAvg/sub/xЭ
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/203186*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/sub’
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_1_assignmovingavg_203186*
_output_shapes
:0*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpЇ
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/203186*
_output_shapes
:02-
+batch_normalization_1/AssignMovingAvg/sub_1£
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/203186*
_output_shapes
:02+
)batch_normalization_1/AssignMovingAvg/mulЕ
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_1_assignmovingavg_203186-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/203186*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/203193*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_1/AssignMovingAvg_1/sub/x•
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/203193*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/subџ
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_1_assignmovingavg_1_203193*
_output_shapes
:0*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp∆
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/203193*
_output_shapes
:02/
-batch_normalization_1/AssignMovingAvg_1/sub_1≠
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/203193*
_output_shapes
:02-
+batch_normalization_1/AssignMovingAvg_1/mulС
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_1_assignmovingavg_1_203193/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/203193*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpЃ
quant_conv2d_2/ste_sign_4/SignSign*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 02 
quant_conv2d_2/ste_sign_4/SignЗ
quant_conv2d_2/ste_sign_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2!
quant_conv2d_2/ste_sign_4/add/yѕ
quant_conv2d_2/ste_sign_4/addAddV2"quant_conv2d_2/ste_sign_4/Sign:y:0(quant_conv2d_2/ste_sign_4/add/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 02
quant_conv2d_2/ste_sign_4/add©
 quant_conv2d_2/ste_sign_4/Sign_1Sign!quant_conv2d_2/ste_sign_4/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€ 02"
 quant_conv2d_2/ste_sign_4/Sign_1і
"quant_conv2d_2/ste_sign_4/IdentityIdentity$quant_conv2d_2/ste_sign_4/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 02$
"quant_conv2d_2/ste_sign_4/Identity∞
#quant_conv2d_2/ste_sign_4/IdentityN	IdentityN$quant_conv2d_2/ste_sign_4/Sign_1:y:0*batch_normalization_1/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-203200*J
_output_shapes8
6:€€€€€€€€€ 0:€€€€€€€€€ 02%
#quant_conv2d_2/ste_sign_4/IdentityN¬
$quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02&
$quant_conv2d_2/Conv2D/ReadVariableOpµ
%quant_conv2d_2/Conv2D/ste_sign_3/SignSign,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:002'
%quant_conv2d_2/Conv2D/ste_sign_3/SignХ
&quant_conv2d_2/Conv2D/ste_sign_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2(
&quant_conv2d_2/Conv2D/ste_sign_3/add/yв
$quant_conv2d_2/Conv2D/ste_sign_3/addAddV2)quant_conv2d_2/Conv2D/ste_sign_3/Sign:y:0/quant_conv2d_2/Conv2D/ste_sign_3/add/y:output:0*
T0*&
_output_shapes
:002&
$quant_conv2d_2/Conv2D/ste_sign_3/addµ
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1Sign(quant_conv2d_2/Conv2D/ste_sign_3/add:z:0*
T0*&
_output_shapes
:002)
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1ј
)quant_conv2d_2/Conv2D/ste_sign_3/IdentityIdentity+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0*
T0*&
_output_shapes
:002+
)quant_conv2d_2/Conv2D/ste_sign_3/Identityµ
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityN	IdentityN+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203210*8
_output_shapes&
$:00:002,
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityNэ
quant_conv2d_2/Conv2DConv2D,quant_conv2d_2/ste_sign_4/IdentityN:output:03quant_conv2d_2/Conv2D/ste_sign_3/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 0*
paddingSAME*
strides
2
quant_conv2d_2/Conv2D 
max_pooling2d_2/MaxPoolMaxPoolquant_conv2d_2/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolК
"batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/xК
"batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/yƒ
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAndґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_2/ReadVariableOp_1}
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_2/ConstБ
batch_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_2/Const_1£
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0$batch_normalization_2/Const:output:0&batch_normalization_2/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2(
&batch_normalization_2/FusedBatchNormV3Г
batch_normalization_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_2/Const_2а
+batch_normalization_2/AssignMovingAvg/sub/xConst*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/203236*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_2/AssignMovingAvg/sub/xЭ
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/203236*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/sub’
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_2_assignmovingavg_203236*
_output_shapes
:0*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpЇ
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/203236*
_output_shapes
:02-
+batch_normalization_2/AssignMovingAvg/sub_1£
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/203236*
_output_shapes
:02+
)batch_normalization_2/AssignMovingAvg/mulЕ
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_2_assignmovingavg_203236-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/203236*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/203243*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_2/AssignMovingAvg_1/sub/x•
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/203243*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/subџ
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_2_assignmovingavg_1_203243*
_output_shapes
:0*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp∆
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/203243*
_output_shapes
:02/
-batch_normalization_2/AssignMovingAvg_1/sub_1≠
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/203243*
_output_shapes
:02-
+batch_normalization_2/AssignMovingAvg_1/mulС
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_2_assignmovingavg_1_203243/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/203243*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpЃ
quant_conv2d_3/ste_sign_6/SignSign*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€02 
quant_conv2d_3/ste_sign_6/SignЗ
quant_conv2d_3/ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2!
quant_conv2d_3/ste_sign_6/add/yѕ
quant_conv2d_3/ste_sign_6/addAddV2"quant_conv2d_3/ste_sign_6/Sign:y:0(quant_conv2d_3/ste_sign_6/add/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€02
quant_conv2d_3/ste_sign_6/add©
 quant_conv2d_3/ste_sign_6/Sign_1Sign!quant_conv2d_3/ste_sign_6/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€02"
 quant_conv2d_3/ste_sign_6/Sign_1і
"quant_conv2d_3/ste_sign_6/IdentityIdentity$quant_conv2d_3/ste_sign_6/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€02$
"quant_conv2d_3/ste_sign_6/Identity∞
#quant_conv2d_3/ste_sign_6/IdentityN	IdentityN$quant_conv2d_3/ste_sign_6/Sign_1:y:0*batch_normalization_2/FusedBatchNormV3:y:0*
T
2*,
_gradient_op_typeCustomGradient-203250*J
_output_shapes8
6:€€€€€€€€€0:€€€€€€€€€02%
#quant_conv2d_3/ste_sign_6/IdentityN¬
$quant_conv2d_3/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype02&
$quant_conv2d_3/Conv2D/ReadVariableOpµ
%quant_conv2d_3/Conv2D/ste_sign_5/SignSign,quant_conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:002'
%quant_conv2d_3/Conv2D/ste_sign_5/SignХ
&quant_conv2d_3/Conv2D/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2(
&quant_conv2d_3/Conv2D/ste_sign_5/add/yв
$quant_conv2d_3/Conv2D/ste_sign_5/addAddV2)quant_conv2d_3/Conv2D/ste_sign_5/Sign:y:0/quant_conv2d_3/Conv2D/ste_sign_5/add/y:output:0*
T0*&
_output_shapes
:002&
$quant_conv2d_3/Conv2D/ste_sign_5/addµ
'quant_conv2d_3/Conv2D/ste_sign_5/Sign_1Sign(quant_conv2d_3/Conv2D/ste_sign_5/add:z:0*
T0*&
_output_shapes
:002)
'quant_conv2d_3/Conv2D/ste_sign_5/Sign_1ј
)quant_conv2d_3/Conv2D/ste_sign_5/IdentityIdentity+quant_conv2d_3/Conv2D/ste_sign_5/Sign_1:y:0*
T0*&
_output_shapes
:002+
)quant_conv2d_3/Conv2D/ste_sign_5/Identityµ
*quant_conv2d_3/Conv2D/ste_sign_5/IdentityN	IdentityN+quant_conv2d_3/Conv2D/ste_sign_5/Sign_1:y:0,quant_conv2d_3/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203260*8
_output_shapes&
$:00:002,
*quant_conv2d_3/Conv2D/ste_sign_5/IdentityNэ
quant_conv2d_3/Conv2DConv2D,quant_conv2d_3/ste_sign_6/IdentityN:output:03quant_conv2d_3/Conv2D/ste_sign_5/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€0*
paddingSAME*
strides
2
quant_conv2d_3/Conv2D 
max_pooling2d_3/MaxPoolMaxPoolquant_conv2d_3/Conv2D:output:0*/
_output_shapes
:€€€€€€€€€0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolК
"batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/xК
"batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/yƒ
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAndґ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:0*
dtype02&
$batch_normalization_3/ReadVariableOpЉ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:0*
dtype02(
&batch_normalization_3/ReadVariableOp_1}
batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_3/ConstБ
batch_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_3/Const_1£
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_3/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0$batch_normalization_3/Const:output:0&batch_normalization_3/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2(
&batch_normalization_3/FusedBatchNormV3Г
batch_normalization_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_3/Const_2а
+batch_normalization_3/AssignMovingAvg/sub/xConst*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/203286*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_3/AssignMovingAvg/sub/xЭ
)batch_normalization_3/AssignMovingAvg/subSub4batch_normalization_3/AssignMovingAvg/sub/x:output:0&batch_normalization_3/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/203286*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/sub’
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_3_assignmovingavg_203286*
_output_shapes
:0*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpЇ
+batch_normalization_3/AssignMovingAvg/sub_1Sub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_3/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/203286*
_output_shapes
:02-
+batch_normalization_3/AssignMovingAvg/sub_1£
)batch_normalization_3/AssignMovingAvg/mulMul/batch_normalization_3/AssignMovingAvg/sub_1:z:0-batch_normalization_3/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/203286*
_output_shapes
:02+
)batch_normalization_3/AssignMovingAvg/mulЕ
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_3_assignmovingavg_203286-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/203286*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/203293*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_3/AssignMovingAvg_1/sub/x•
+batch_normalization_3/AssignMovingAvg_1/subSub6batch_normalization_3/AssignMovingAvg_1/sub/x:output:0&batch_normalization_3/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/203293*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/subџ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_3_assignmovingavg_1_203293*
_output_shapes
:0*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp∆
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_3/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/203293*
_output_shapes
:02/
-batch_normalization_3/AssignMovingAvg_1/sub_1≠
+batch_normalization_3/AssignMovingAvg_1/mulMul1batch_normalization_3/AssignMovingAvg_1/sub_1:z:0/batch_normalization_3/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/203293*
_output_shapes
:02-
+batch_normalization_3/AssignMovingAvg_1/mulС
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_3_assignmovingavg_1_203293/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/203293*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
flatten/Const§
flatten/ReshapeReshape*batch_normalization_3/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/ReshapeП
quant_dense/ste_sign_8/SignSignflatten/Reshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_8/SignБ
quant_dense/ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
quant_dense/ste_sign_8/add/yЉ
quant_dense/ste_sign_8/addAddV2quant_dense/ste_sign_8/Sign:y:0%quant_dense/ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_8/addЩ
quant_dense/ste_sign_8/Sign_1Signquant_dense/ste_sign_8/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_8/Sign_1§
quant_dense/ste_sign_8/IdentityIdentity!quant_dense/ste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
quant_dense/ste_sign_8/IdentityЗ
 quant_dense/ste_sign_8/IdentityN	IdentityN!quant_dense/ste_sign_8/Sign_1:y:0flatten/Reshape:output:0*
T
2*,
_gradient_op_typeCustomGradient-203302*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2"
 quant_dense/ste_sign_8/IdentityN≥
!quant_dense/MatMul/ReadVariableOpReadVariableOp*quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!quant_dense/MatMul/ReadVariableOp¶
"quant_dense/MatMul/ste_sign_7/SignSign)quant_dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"quant_dense/MatMul/ste_sign_7/SignП
#quant_dense/MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2%
#quant_dense/MatMul/ste_sign_7/add/y–
!quant_dense/MatMul/ste_sign_7/addAddV2&quant_dense/MatMul/ste_sign_7/Sign:y:0,quant_dense/MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
АА2#
!quant_dense/MatMul/ste_sign_7/add¶
$quant_dense/MatMul/ste_sign_7/Sign_1Sign%quant_dense/MatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
АА2&
$quant_dense/MatMul/ste_sign_7/Sign_1±
&quant_dense/MatMul/ste_sign_7/IdentityIdentity(quant_dense/MatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
АА2(
&quant_dense/MatMul/ste_sign_7/IdentityЭ
'quant_dense/MatMul/ste_sign_7/IdentityN	IdentityN(quant_dense/MatMul/ste_sign_7/Sign_1:y:0)quant_dense/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203312*,
_output_shapes
:
АА:
АА2)
'quant_dense/MatMul/ste_sign_7/IdentityN¬
quant_dense/MatMulMatMul)quant_dense/ste_sign_8/IdentityN:output:00quant_dense/MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/MatMulК
"batch_normalization_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_4/LogicalAnd/xК
"batch_normalization_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_4/LogicalAnd/yƒ
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_4/LogicalAndґ
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_4/moments/mean/reduction_indicesи
"batch_normalization_4/moments/meanMeanquant_dense/MatMul:product:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2$
"batch_normalization_4/moments/meanњ
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	А2,
*batch_normalization_4/moments/StopGradientэ
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencequant_dense/MatMul:product:03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€А21
/batch_normalization_4/moments/SquaredDifferenceЊ
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_4/moments/variance/reduction_indicesЛ
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2(
&batch_normalization_4/moments/variance√
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_4/moments/SqueezeЋ
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1а
+batch_normalization_4/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/203332*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2-
+batch_normalization_4/AssignMovingAvg/decay÷
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_4_assignmovingavg_203332*
_output_shapes	
:А*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp≤
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/203332*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/sub©
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/203332*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/mulЕ
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_4_assignmovingavg_203332-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/203332*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_4/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/203338*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2/
-batch_normalization_4/AssignMovingAvg_1/decay№
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_4_assignmovingavg_1_203338*
_output_shapes	
:А*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpЉ
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/203338*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/sub≥
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/203338*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/mulС
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_4_assignmovingavg_1_203338/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/203338*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/yџ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/add¶
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/Rsqrtб
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpё
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/mulѕ
%batch_normalization_4/batchnorm/mul_1Mulquant_dense/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_4/batchnorm/mul_1‘
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/mul_2’
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpЏ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/subё
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_4/batchnorm/add_1¶
quant_dense_1/ste_sign_10/SignSign)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
quant_dense_1/ste_sign_10/SignЗ
quant_dense_1/ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2!
quant_dense_1/ste_sign_10/add/y»
quant_dense_1/ste_sign_10/addAddV2"quant_dense_1/ste_sign_10/Sign:y:0(quant_dense_1/ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense_1/ste_sign_10/addҐ
 quant_dense_1/ste_sign_10/Sign_1Sign!quant_dense_1/ste_sign_10/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 quant_dense_1/ste_sign_10/Sign_1≠
"quant_dense_1/ste_sign_10/IdentityIdentity$quant_dense_1/ste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"quant_dense_1/ste_sign_10/Identity°
#quant_dense_1/ste_sign_10/IdentityN	IdentityN$quant_dense_1/ste_sign_10/Sign_1:y:0)batch_normalization_4/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-203356*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2%
#quant_dense_1/ste_sign_10/IdentityNЄ
#quant_dense_1/MatMul/ReadVariableOpReadVariableOp,quant_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02%
#quant_dense_1/MatMul/ReadVariableOpЂ
$quant_dense_1/MatMul/ste_sign_9/SignSign+quant_dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2&
$quant_dense_1/MatMul/ste_sign_9/SignУ
%quant_dense_1/MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2'
%quant_dense_1/MatMul/ste_sign_9/add/y„
#quant_dense_1/MatMul/ste_sign_9/addAddV2(quant_dense_1/MatMul/ste_sign_9/Sign:y:0.quant_dense_1/MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	А2%
#quant_dense_1/MatMul/ste_sign_9/addЂ
&quant_dense_1/MatMul/ste_sign_9/Sign_1Sign'quant_dense_1/MatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	А2(
&quant_dense_1/MatMul/ste_sign_9/Sign_1ґ
(quant_dense_1/MatMul/ste_sign_9/IdentityIdentity*quant_dense_1/MatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	А2*
(quant_dense_1/MatMul/ste_sign_9/Identity£
)quant_dense_1/MatMul/ste_sign_9/IdentityN	IdentityN*quant_dense_1/MatMul/ste_sign_9/Sign_1:y:0+quant_dense_1/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-203366**
_output_shapes
:	А:	А2+
)quant_dense_1/MatMul/ste_sign_9/IdentityN 
quant_dense_1/MatMulMatMul,quant_dense_1/ste_sign_10/IdentityN:output:02quant_dense_1/MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
quant_dense_1/MatMulК
"batch_normalization_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_5/LogicalAnd/xК
"batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_5/LogicalAnd/yƒ
 batch_normalization_5/LogicalAnd
LogicalAnd+batch_normalization_5/LogicalAnd/x:output:0+batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_5/LogicalAndґ
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_5/moments/mean/reduction_indicesй
"batch_normalization_5/moments/meanMeanquant_dense_1/MatMul:product:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_5/moments/meanЊ
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_5/moments/StopGradientю
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencequant_dense_1/MatMul:product:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€21
/batch_normalization_5/moments/SquaredDifferenceЊ
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_5/moments/variance/reduction_indicesК
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_5/moments/variance¬
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_5/moments/Squeeze 
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1а
+batch_normalization_5/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/203386*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2-
+batch_normalization_5/AssignMovingAvg/decay’
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_5_assignmovingavg_203386*
_output_shapes
:*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOp±
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/203386*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/sub®
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/203386*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/mulЕ
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_5_assignmovingavg_203386-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/203386*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_5/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/203392*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2/
-batch_normalization_5/AssignMovingAvg_1/decayџ
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_5_assignmovingavg_1_203392*
_output_shapes
:*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpї
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/203392*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/sub≤
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/203392*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/mulС
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_5_assignmovingavg_1_203392/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/203392*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_5/batchnorm/add/yЏ
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/add•
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtа
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mul–
%batch_normalization_5/batchnorm/mul_1Mulquant_dense_1/MatMul:product:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%batch_normalization_5/batchnorm/mul_1”
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2‘
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpў
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/subЁ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%batch_normalization_5/batchnorm/add_1Р
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
activation/Softmax–
IdentityIdentityactivation/Softmax:softmax:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp#^quant_conv2d/Conv2D/ReadVariableOp%^quant_conv2d_1/Conv2D/ReadVariableOp%^quant_conv2d_2/Conv2D/ReadVariableOp%^quant_conv2d_3/Conv2D/ReadVariableOp"^quant_dense/MatMul/ReadVariableOp$^quant_dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::2r
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
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12v
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
$quant_conv2d_2/Conv2D/ReadVariableOp$quant_conv2d_2/Conv2D/ReadVariableOp2L
$quant_conv2d_3/Conv2D/ReadVariableOp$quant_conv2d_3/Conv2D/ReadVariableOp2F
!quant_dense/MatMul/ReadVariableOp!quant_dense/MatMul/ReadVariableOp2J
#quant_dense_1/MatMul/ReadVariableOp#quant_dense_1/MatMul/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€В
 
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
И
©
6__inference_batch_normalization_3_layer_call_fn_204460

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2022772
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€0
 
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
≠%
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_202182

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_202167
assignmovingavg_1_202174
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/202167*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/202167*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_202167*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/202167*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/202167*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_202167AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/202167*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/202174*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202174*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_202174*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202174*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202174*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_202174AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/202174*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€0
 
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
К
u
/__inference_quant_conv2d_1_layer_call_fn_201075

inputs
unknown
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2010672
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs:

_output_shapes
: 
Њ\
Ј
F__inference_sequential_layer_call_and_return_conditional_losses_202817

inputs
quant_conv2d_202738
batch_normalization_202742
batch_normalization_202744
batch_normalization_202746
batch_normalization_202748
quant_conv2d_1_202751 
batch_normalization_1_202755 
batch_normalization_1_202757 
batch_normalization_1_202759 
batch_normalization_1_202761
quant_conv2d_2_202764 
batch_normalization_2_202768 
batch_normalization_2_202770 
batch_normalization_2_202772 
batch_normalization_2_202774
quant_conv2d_3_202777 
batch_normalization_3_202781 
batch_normalization_3_202783 
batch_normalization_3_202785 
batch_normalization_3_202787
quant_dense_202791 
batch_normalization_4_202794 
batch_normalization_4_202796 
batch_normalization_4_202798 
batch_normalization_4_202800
quant_dense_1_202803 
batch_normalization_5_202806 
batch_normalization_5_202808 
batch_normalization_5_202810 
batch_normalization_5_202812
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ$quant_conv2d/StatefulPartitionedCallҐ&quant_conv2d_1/StatefulPartitionedCallҐ&quant_conv2d_2/StatefulPartitionedCallҐ&quant_conv2d_3/StatefulPartitionedCallҐ#quant_dense/StatefulPartitionedCallҐ%quant_dense_1/StatefulPartitionedCall÷
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsquant_conv2d_202738*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В0*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_2008572&
$quant_conv2d/StatefulPartitionedCall—
max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2008712
max_pooling2d/PartitionedCallи
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_202742batch_normalization_202744batch_normalization_202746batch_normalization_202748*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2020142-
+batch_normalization/StatefulPartitionedCallЛ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0quant_conv2d_1_202751*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2010672(
&quant_conv2d_1/StatefulPartitionedCallў
max_pooling2d_1/PartitionedCallPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2010812!
max_pooling2d_1/PartitionedCallш
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_202755batch_normalization_1_202757batch_normalization_1_202759batch_normalization_1_202761*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2021092/
-batch_normalization_1/StatefulPartitionedCallН
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0quant_conv2d_2_202764*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2012772(
&quant_conv2d_2/StatefulPartitionedCallў
max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2012912!
max_pooling2d_2/PartitionedCallш
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_202768batch_normalization_2_202770batch_normalization_2_202772batch_normalization_2_202774*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2022042/
-batch_normalization_2/StatefulPartitionedCallН
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0quant_conv2d_3_202777*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2014872(
&quant_conv2d_3/StatefulPartitionedCallў
max_pooling2d_3/PartitionedCallPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2015012!
max_pooling2d_3/PartitionedCallш
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_202781batch_normalization_3_202783batch_normalization_3_202785batch_normalization_3_202787*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2022992/
-batch_normalization_3/StatefulPartitionedCallЅ
flatten/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2023412
flatten/PartitionedCallд
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_202791*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_2023722%
#quant_dense/StatefulPartitionedCallх
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_4_202794batch_normalization_4_202796batch_normalization_4_202798batch_normalization_4_202800*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2017882/
-batch_normalization_4/StatefulPartitionedCallБ
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_1_202803*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_2024422'
%quant_dense_1/StatefulPartitionedCallц
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_5_202806batch_normalization_5_202808batch_normalization_5_202810batch_normalization_5_202812*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2019402/
-batch_normalization_5/StatefulPartitionedCall…
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_2024942
activation/PartitionedCallЕ
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€В
 
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
И
И
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_201788

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOp^
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

LogicalAndУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/add_1№
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
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
х%
Ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_201601

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_201586
assignmovingavg_1_201593
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/201586*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/201586*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_201586*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/201586*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/201586*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_201586AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/201586*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/201593*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201593*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_201593*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201593*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201593*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_201593AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/201593*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
Љ
b
F__inference_activation_layer_call_and_return_conditional_losses_202494

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
—
d
F__inference_ste_sign_5_layer_call_and_return_conditional_losses_204899

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:002
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:002
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:002
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:002

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-204890*8
_output_shapes&
$:00:002
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:002

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:00:N J
&
_output_shapes
:00
 
_user_specified_nameinputs
≠%
Ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_202277

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_202262
assignmovingavg_1_202269
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/202262*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/202262*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_202262*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/202262*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/202262*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_202262AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/202262*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/202269*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202269*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_202269*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202269*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/202269*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_202269AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/202269*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€0
 
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
М
Ґ
G__inference_quant_dense_layer_call_and_return_conditional_losses_202372

inputs"
matmul_readvariableop_resource
identityИҐMatMul/ReadVariableOpe
ste_sign_8/SignSigninputs*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_8/Signi
ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
ste_sign_8/add/yМ
ste_sign_8/addAddV2ste_sign_8/Sign:y:0ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_8/addu
ste_sign_8/Sign_1Signste_sign_8/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_8/Sign_1А
ste_sign_8/IdentityIdentityste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_8/Identity—
ste_sign_8/IdentityN	IdentityNste_sign_8/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-202352*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2
ste_sign_8/IdentityNП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpВ
MatMul/ste_sign_7/SignSignMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_7/Signw
MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
MatMul/ste_sign_7/add/y†
MatMul/ste_sign_7/addAddV2MatMul/ste_sign_7/Sign:y:0 MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_7/addВ
MatMul/ste_sign_7/Sign_1SignMatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_7/Sign_1Н
MatMul/ste_sign_7/IdentityIdentityMatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_7/Identityн
MatMul/ste_sign_7/IdentityN	IdentityNMatMul/ste_sign_7/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-202362*,
_output_shapes
:
АА:
АА2
MatMul/ste_sign_7/IdentityNТ
MatMulMatMulste_sign_8/IdentityN:output:0$MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€А:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: 
ѕ
b
D__inference_ste_sign_layer_call_and_return_conditional_losses_204814

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:02
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:02
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:02
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:02

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-204805*8
_output_shapes&
$:0:02
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:02

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:0:N J
&
_output_shapes
:0
 
_user_specified_nameinputs
Њ\
Ј
F__inference_sequential_layer_call_and_return_conditional_losses_202670

inputs
quant_conv2d_202591
batch_normalization_202595
batch_normalization_202597
batch_normalization_202599
batch_normalization_202601
quant_conv2d_1_202604 
batch_normalization_1_202608 
batch_normalization_1_202610 
batch_normalization_1_202612 
batch_normalization_1_202614
quant_conv2d_2_202617 
batch_normalization_2_202621 
batch_normalization_2_202623 
batch_normalization_2_202625 
batch_normalization_2_202627
quant_conv2d_3_202630 
batch_normalization_3_202634 
batch_normalization_3_202636 
batch_normalization_3_202638 
batch_normalization_3_202640
quant_dense_202644 
batch_normalization_4_202647 
batch_normalization_4_202649 
batch_normalization_4_202651 
batch_normalization_4_202653
quant_dense_1_202656 
batch_normalization_5_202659 
batch_normalization_5_202661 
batch_normalization_5_202663 
batch_normalization_5_202665
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ$quant_conv2d/StatefulPartitionedCallҐ&quant_conv2d_1/StatefulPartitionedCallҐ&quant_conv2d_2/StatefulPartitionedCallҐ&quant_conv2d_3/StatefulPartitionedCallҐ#quant_dense/StatefulPartitionedCallҐ%quant_dense_1/StatefulPartitionedCall÷
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsquant_conv2d_202591*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В0*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_2008572&
$quant_conv2d/StatefulPartitionedCall—
max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2008712
max_pooling2d/PartitionedCallи
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_202595batch_normalization_202597batch_normalization_202599batch_normalization_202601*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2019922-
+batch_normalization/StatefulPartitionedCallЛ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0quant_conv2d_1_202604*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_2010672(
&quant_conv2d_1/StatefulPartitionedCallў
max_pooling2d_1/PartitionedCallPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2010812!
max_pooling2d_1/PartitionedCallш
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_202608batch_normalization_1_202610batch_normalization_1_202612batch_normalization_1_202614*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2020872/
-batch_normalization_1/StatefulPartitionedCallН
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0quant_conv2d_2_202617*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2012772(
&quant_conv2d_2/StatefulPartitionedCallў
max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2012912!
max_pooling2d_2/PartitionedCallш
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_202621batch_normalization_2_202623batch_normalization_2_202625batch_normalization_2_202627*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2021822/
-batch_normalization_2/StatefulPartitionedCallН
&quant_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0quant_conv2d_3_202630*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_2014872(
&quant_conv2d_3/StatefulPartitionedCallў
max_pooling2d_3/PartitionedCallPartitionedCall/quant_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2015012!
max_pooling2d_3/PartitionedCallш
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_202634batch_normalization_3_202636batch_normalization_3_202638batch_normalization_3_202640*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2022772/
-batch_normalization_3/StatefulPartitionedCallЅ
flatten/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2023412
flatten/PartitionedCallд
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_202644*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_2023722%
#quant_dense/StatefulPartitionedCallх
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_4_202647batch_normalization_4_202649batch_normalization_4_202651batch_normalization_4_202653*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2017522/
-batch_normalization_4/StatefulPartitionedCallБ
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_1_202656*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_2024422'
%quant_dense_1/StatefulPartitionedCallц
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_5_202659batch_normalization_5_202661batch_normalization_5_202663batch_normalization_5_202665*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2019042/
-batch_normalization_5/StatefulPartitionedCall…
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_2024942
activation/PartitionedCallЕ
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall'^quant_conv2d_3/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2P
&quant_conv2d_1/StatefulPartitionedCall&quant_conv2d_1/StatefulPartitionedCall2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2P
&quant_conv2d_3/StatefulPartitionedCall&quant_conv2d_3/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€В
 
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
И
©
6__inference_batch_normalization_1_layer_call_fn_204121

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2021092
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 0
 
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
Љ
ф
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204365

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constм
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
ї
_
C__inference_flatten_layer_call_and_return_conditional_losses_204479

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€0:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
ф
ф
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204447

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstЏ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€0
 
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
—
d
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_201266

inputs

identity_1M
SignSigninputs*
T0*&
_output_shapes
:002
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/y^
addAddV2Sign:y:0add/y:output:0*
T0*&
_output_shapes
:002
addR
Sign_1Signadd:z:0*
T0*&
_output_shapes
:002
Sign_1]
IdentityIdentity
Sign_1:y:0*
T0*&
_output_shapes
:002

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-201257*8
_output_shapes&
$:00:002
	IdentityNi

Identity_1IdentityIdentityN:output:0*
T0*&
_output_shapes
:002

Identity_1"!

identity_1Identity_1:output:0*%
_input_shapes
:00:N J
&
_output_shapes
:00
 
_user_specified_nameinputs
м
L
0__inference_max_pooling2d_2_layer_call_fn_201297

inputs
identityЂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2012912
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ї
_
C__inference_flatten_layer_call_and_return_conditional_losses_202341

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€0:W S
/
_output_shapes
:€€€€€€€€€0
 
_user_specified_nameinputs
–
©
6__inference_batch_normalization_1_layer_call_fn_204026

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2011812
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
Б
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_201501

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
т
т
O__inference_batch_normalization_layer_call_and_return_conditional_losses_202014

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstЏ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€A
02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€A
0
 
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
х%
Ь
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_201181

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_201166
assignmovingavg_1_201173
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/201166*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/201166*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_201166*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/201166*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/201166*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_201166AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/201166*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/201173*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201173*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_201173*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201173*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201173*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_201173AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/201173*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
г
√
+__inference_sequential_layer_call_fn_202880
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
identityИҐStatefulPartitionedCall†
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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2028172
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*©
_input_shapesЧ
Ф:€€€€€€€€€В::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:€€€€€€€€€В
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
Ђ%
Ъ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_201992

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_201977
assignmovingavg_1_201984
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/201977*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/201977*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_201977*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/201977*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/201977*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_201977AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/201977*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/201984*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201984*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_201984*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201984*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/201984*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_201984AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/201984*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€A
02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€A
0
 
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
≠%
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204167

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_204152
assignmovingavg_1_204159
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *fff?2	
Const_2Ю
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/204152*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/204152*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_204152*
_output_shapes
:0*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/204152*
_output_shapes
:02
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/204152*
_output_shapes
:02
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_204152AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/204152*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/204159*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204159*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_204159*
_output_shapes
:0*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204159*
_output_shapes
:02
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/204159*
_output_shapes
:02
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_204159AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/204159*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€0
 
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
≈	
d
F__inference_ste_sign_6_layer_call_and_return_conditional_losses_201453

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/yy
addAddV2Sign:y:0add/y:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identityв
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-201444*n
_output_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
	IdentityNД

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
’
G
+__inference_activation_layer_call_fn_204802

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_2024942
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ф
ф
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_202204

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
ConstЏ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€0
 
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
Ї
т
O__inference_batch_normalization_layer_call_and_return_conditional_losses_201006

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1^
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
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constм
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
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
К
u
/__inference_quant_conv2d_2_layer_call_fn_201285

inputs
unknown
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_2012772
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs:

_output_shapes
: 
≈	
d
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_204848

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
SignS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
add/yy
addAddV2Sign:y:0add/y:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identityв
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-204839*n
_output_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02
	IdentityNД

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
и
J
.__inference_max_pooling2d_layer_call_fn_200877

inputs
identity©
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2008712
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ж
s
-__inference_quant_conv2d_layer_call_fn_200865

inputs
unknown
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_2008572
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€02

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: 
м
L
0__inference_max_pooling2d_1_layer_call_fn_201087

inputs
identityЂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2010812
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
м
©
6__inference_batch_normalization_4_layer_call_fn_204625

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2017522
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
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
Ґ
t
.__inference_quant_dense_1_layer_call_fn_204668

inputs
unknown
identityИҐStatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_2024422
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€А:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: "ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ћ
serving_defaultЄ
Z
quant_conv2d_inputD
$serving_default_quant_conv2d_input:0€€€€€€€€€В>

activation0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:°Э
Ќ£
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+ы&call_and_return_all_conditional_losses
ь__call__
э_default_save_signature"“Э
_tf_keras_sequential≤Э{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "QuantConv2D", "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "QuantDense", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}], "build_input_shape": [null, 130, 20, 1]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "QuantConv2D", "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "QuantDense", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}], "build_input_shape": [null, 130, 20, 1]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": ["accuracy", "sparse_categorical_accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0003162277571391314, "decay": 0.0, "beta_1": 0.949999988079071, "beta_2": 0.9990000128746033, "epsilon": 1e-08, "amsgrad": false}}}}
Ђ

kernel_quantizer

kernel
	variables
regularization_losses
trainable_variables
	keras_api
+ю&call_and_return_all_conditional_losses
€__call__"ш
_tf_keras_layerё{"class_name": "QuantConv2D", "name": "quant_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 130, 20, 1], "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
ы
	variables
 regularization_losses
!trainable_variables
"	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"к
_tf_keras_layer–{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
∞
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(	variables
)regularization_losses
*trainable_variables
+	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"Џ
_tf_keras_layerј{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}}
Б
,kernel_quantizer
-input_quantizer

.kernel
/	variables
0regularization_losses
1trainable_variables
2	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"є	
_tf_keras_layerЯ	{"class_name": "QuantConv2D", "name": "quant_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}}}
€
3	variables
4regularization_losses
5trainable_variables
6	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"о
_tf_keras_layer‘{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
і
7axis
	8gamma
9beta
:moving_mean
;moving_variance
<	variables
=regularization_losses
>trainable_variables
?	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"ё
_tf_keras_layerƒ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}}
Б
@kernel_quantizer
Ainput_quantizer

Bkernel
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"є	
_tf_keras_layerЯ	{"class_name": "QuantConv2D", "name": "quant_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}}}
€
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"о
_tf_keras_layer‘{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
і
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
+О&call_and_return_all_conditional_losses
П__call__"ё
_tf_keras_layerƒ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}}
Б
Tkernel_quantizer
Uinput_quantizer

Vkernel
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"є	
_tf_keras_layerЯ	{"class_name": "QuantConv2D", "name": "quant_conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_conv2d_3", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}}}
€
[	variables
\regularization_losses
]trainable_variables
^	keras_api
+Т&call_and_return_all_conditional_losses
У__call__"о
_tf_keras_layer‘{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
і
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
+Ф&call_and_return_all_conditional_losses
Х__call__"ё
_tf_keras_layerƒ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}}
Ѓ
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"Э
_tf_keras_layerГ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
с	
lkernel_quantizer
minput_quantizer

nkernel
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"©
_tf_keras_layerП{"class_name": "QuantDense", "name": "quant_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}}
µ
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"я
_tf_keras_layer≈{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 384}}}}
ч	
|kernel_quantizer
}input_quantizer

~kernel
	variables
Аregularization_losses
Бtrainable_variables
В	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"ђ
_tf_keras_layerТ{"class_name": "QuantDense", "name": "quant_dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}}
Љ
	Гaxis

Дgamma
	Еbeta
Жmoving_mean
Зmoving_variance
И	variables
Йregularization_losses
Кtrainable_variables
Л	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"Ё
_tf_keras_layer√{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 2}}}}
§
М	variables
Нregularization_losses
Оtrainable_variables
П	keras_api
+†&call_and_return_all_conditional_losses
°__call__"П
_tf_keras_layerх{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}
ƒ
	Рiter
Сbeta_1
Тbeta_2

Уdecay
Фlearning_ratem„$mЎ%mў.mЏ8mџ9m№BmЁLmёMmяVmа`mбamвnmгtmдumе~mж	Дmз	Еmиvй$vк%vл.vм8vн9vоBvпLvрMvсVvт`vуavфnvхtvцuvч~vш	Дvщ	Еvъ"
	optimizer
К
0
$1
%2
&3
'4
.5
86
97
:8
;9
B10
L11
M12
N13
O14
V15
`16
a17
b18
c19
n20
t21
u22
v23
w24
~25
Д26
Е27
Ж28
З29"
trackable_list_wrapper
 "
trackable_list_wrapper
®
0
$1
%2
.3
84
95
B6
L7
M8
V9
`10
a11
n12
t13
u14
~15
Д16
Е17"
trackable_list_wrapper
њ
Хmetrics
Цlayers
 Чlayer_regularization_losses
Шnon_trainable_variables
	variables
regularization_losses
trainable_variables
ь__call__
э_default_save_signature
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
-
Ґserving_default"
signature_map
≠
Щ_custom_metrics
Ъ	variables
Ыregularization_losses
Ьtrainable_variables
Э	keras_api
+£&call_and_return_all_conditional_losses
§__call__"В
_tf_keras_layerи{"class_name": "SteSign", "name": "ste_sign", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
-:+02quant_conv2d/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
Юmetrics
Яlayers
 †layer_regularization_losses
°non_trainable_variables
	variables
regularization_losses
trainable_variables
€__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Ґmetrics
£layers
 §layer_regularization_losses
•non_trainable_variables
	variables
 regularization_losses
!trainable_variables
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%02batch_normalization/gamma
&:$02batch_normalization/beta
/:-0 (2batch_normalization/moving_mean
3:10 (2#batch_normalization/moving_variance
<
$0
%1
&2
'3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
°
¶metrics
Іlayers
 ®layer_regularization_losses
©non_trainable_variables
(	variables
)regularization_losses
*trainable_variables
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
±
™_custom_metrics
Ђ	variables
ђregularization_losses
≠trainable_variables
Ѓ	keras_api
+•&call_and_return_all_conditional_losses
¶__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
Ы
ѓ	variables
∞regularization_losses
±trainable_variables
≤	keras_api
+І&call_and_return_all_conditional_losses
®__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
/:-002quant_conv2d_1/kernel
'
.0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
°
≥metrics
іlayers
 µlayer_regularization_losses
ґnon_trainable_variables
/	variables
0regularization_losses
1trainable_variables
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Јmetrics
Єlayers
 єlayer_regularization_losses
Їnon_trainable_variables
3	variables
4regularization_losses
5trainable_variables
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'02batch_normalization_1/gamma
(:&02batch_normalization_1/beta
1:/0 (2!batch_normalization_1/moving_mean
5:30 (2%batch_normalization_1/moving_variance
<
80
91
:2
;3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
°
їmetrics
Љlayers
 љlayer_regularization_losses
Њnon_trainable_variables
<	variables
=regularization_losses
>trainable_variables
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
±
њ_custom_metrics
ј	variables
Ѕregularization_losses
¬trainable_variables
√	keras_api
+©&call_and_return_all_conditional_losses
™__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
Ы
ƒ	variables
≈regularization_losses
∆trainable_variables
«	keras_api
+Ђ&call_and_return_all_conditional_losses
ђ__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
/:-002quant_conv2d_2/kernel
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
B0"
trackable_list_wrapper
°
»metrics
…layers
  layer_regularization_losses
Ћnon_trainable_variables
C	variables
Dregularization_losses
Etrainable_variables
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
ћmetrics
Ќlayers
 ќlayer_regularization_losses
ѕnon_trainable_variables
G	variables
Hregularization_losses
Itrainable_variables
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'02batch_normalization_2/gamma
(:&02batch_normalization_2/beta
1:/0 (2!batch_normalization_2/moving_mean
5:30 (2%batch_normalization_2/moving_variance
<
L0
M1
N2
O3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
°
–metrics
—layers
 “layer_regularization_losses
”non_trainable_variables
P	variables
Qregularization_losses
Rtrainable_variables
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
±
‘_custom_metrics
’	variables
÷regularization_losses
„trainable_variables
Ў	keras_api
+≠&call_and_return_all_conditional_losses
Ѓ__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
Ы
ў	variables
Џregularization_losses
џtrainable_variables
№	keras_api
+ѓ&call_and_return_all_conditional_losses
∞__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
/:-002quant_conv2d_3/kernel
'
V0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
V0"
trackable_list_wrapper
°
Ёmetrics
ёlayers
 яlayer_regularization_losses
аnon_trainable_variables
W	variables
Xregularization_losses
Ytrainable_variables
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
бmetrics
вlayers
 гlayer_regularization_losses
дnon_trainable_variables
[	variables
\regularization_losses
]trainable_variables
У__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'02batch_normalization_3/gamma
(:&02batch_normalization_3/beta
1:/0 (2!batch_normalization_3/moving_mean
5:30 (2%batch_normalization_3/moving_variance
<
`0
a1
b2
c3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
°
еmetrics
жlayers
 зlayer_regularization_losses
иnon_trainable_variables
d	variables
eregularization_losses
ftrainable_variables
Х__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
йmetrics
кlayers
 лlayer_regularization_losses
мnon_trainable_variables
h	variables
iregularization_losses
jtrainable_variables
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
±
н_custom_metrics
о	variables
пregularization_losses
рtrainable_variables
с	keras_api
+±&call_and_return_all_conditional_losses
≤__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
Ы
т	variables
уregularization_losses
фtrainable_variables
х	keras_api
+≥&call_and_return_all_conditional_losses
і__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
&:$
АА2quant_dense/kernel
'
n0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
n0"
trackable_list_wrapper
°
цmetrics
чlayers
 шlayer_regularization_losses
щnon_trainable_variables
o	variables
pregularization_losses
qtrainable_variables
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_4/gamma
):'А2batch_normalization_4/beta
2:0А (2!batch_normalization_4/moving_mean
6:4А (2%batch_normalization_4/moving_variance
<
t0
u1
v2
w3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
°
ъmetrics
ыlayers
 ьlayer_regularization_losses
эnon_trainable_variables
x	variables
yregularization_losses
ztrainable_variables
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
±
ю_custom_metrics
€	variables
Аregularization_losses
Бtrainable_variables
В	keras_api
+µ&call_and_return_all_conditional_losses
ґ__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
Э
Г	variables
Дregularization_losses
Еtrainable_variables
Ж	keras_api
+Ј&call_and_return_all_conditional_losses
Є__call__"И
_tf_keras_layerо{"class_name": "SteSign", "name": "ste_sign_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
':%	А2quant_dense_1/kernel
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
£
Зmetrics
Иlayers
 Йlayer_regularization_losses
Кnon_trainable_variables
	variables
Аregularization_losses
Бtrainable_variables
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
@
Д0
Е1
Ж2
З3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Д0
Е1"
trackable_list_wrapper
§
Лmetrics
Мlayers
 Нlayer_regularization_losses
Оnon_trainable_variables
И	variables
Йregularization_losses
Кtrainable_variables
Я__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Пmetrics
Рlayers
 Сlayer_regularization_losses
Тnon_trainable_variables
М	variables
Нregularization_losses
Оtrainable_variables
°__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
У0
Ф1"
trackable_list_wrapper
¶
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
17"
trackable_list_wrapper
 "
trackable_list_wrapper
x
&0
'1
:2
;3
N4
O5
b6
c7
v8
w9
Ж10
З11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Хmetrics
Цlayers
 Чlayer_regularization_losses
Шnon_trainable_variables
Ъ	variables
Ыregularization_losses
Ьtrainable_variables
§__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Щmetrics
Ъlayers
 Ыlayer_regularization_losses
Ьnon_trainable_variables
Ђ	variables
ђregularization_losses
≠trainable_variables
¶__call__
+•&call_and_return_all_conditional_losses
'•"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Эmetrics
Юlayers
 Яlayer_regularization_losses
†non_trainable_variables
ѓ	variables
∞regularization_losses
±trainable_variables
®__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
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
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
°metrics
Ґlayers
 £layer_regularization_losses
§non_trainable_variables
ј	variables
Ѕregularization_losses
¬trainable_variables
™__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
•metrics
¶layers
 Іlayer_regularization_losses
®non_trainable_variables
ƒ	variables
≈regularization_losses
∆trainable_variables
ђ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
@0
A1"
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
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
©metrics
™layers
 Ђlayer_regularization_losses
ђnon_trainable_variables
’	variables
÷regularization_losses
„trainable_variables
Ѓ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
≠metrics
Ѓlayers
 ѓlayer_regularization_losses
∞non_trainable_variables
ў	variables
Џregularization_losses
џtrainable_variables
∞__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
T0
U1"
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
b0
c1"
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
§
±metrics
≤layers
 ≥layer_regularization_losses
іnon_trainable_variables
о	variables
пregularization_losses
рtrainable_variables
≤__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
µmetrics
ґlayers
 Јlayer_regularization_losses
Єnon_trainable_variables
т	variables
уregularization_losses
фtrainable_variables
і__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
l0
m1"
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
§
єmetrics
Їlayers
 їlayer_regularization_losses
Љnon_trainable_variables
€	variables
Аregularization_losses
Бtrainable_variables
ґ__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
љmetrics
Њlayers
 њlayer_regularization_losses
јnon_trainable_variables
Г	variables
Дregularization_losses
Еtrainable_variables
Є__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
|0
}1"
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
0
Ж0
З1"
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

Ѕtotal

¬count
√
_fn_kwargs
ƒ	variables
≈regularization_losses
∆trainable_variables
«	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"е
_tf_keras_layerЋ{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
…

»total

…count
 
_fn_kwargs
Ћ	variables
ћregularization_losses
Ќtrainable_variables
ќ	keras_api
+ї&call_and_return_all_conditional_losses
Љ__call__"Л
_tf_keras_layerс{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}
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
0
Ѕ0
¬1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
ѕmetrics
–layers
 —layer_regularization_losses
“non_trainable_variables
ƒ	variables
≈regularization_losses
∆trainable_variables
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
»0
…1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
”metrics
‘layers
 ’layer_regularization_losses
÷non_trainable_variables
Ћ	variables
ћregularization_losses
Ќtrainable_variables
Љ__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ѕ0
¬1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
»0
…1"
trackable_list_wrapper
2:002Adam/quant_conv2d/kernel/m
,:*02 Adam/batch_normalization/gamma/m
+:)02Adam/batch_normalization/beta/m
4:2002Adam/quant_conv2d_1/kernel/m
.:,02"Adam/batch_normalization_1/gamma/m
-:+02!Adam/batch_normalization_1/beta/m
4:2002Adam/quant_conv2d_2/kernel/m
.:,02"Adam/batch_normalization_2/gamma/m
-:+02!Adam/batch_normalization_2/beta/m
4:2002Adam/quant_conv2d_3/kernel/m
.:,02"Adam/batch_normalization_3/gamma/m
-:+02!Adam/batch_normalization_3/beta/m
+:)
АА2Adam/quant_dense/kernel/m
/:-А2"Adam/batch_normalization_4/gamma/m
.:,А2!Adam/batch_normalization_4/beta/m
,:*	А2Adam/quant_dense_1/kernel/m
.:,2"Adam/batch_normalization_5/gamma/m
-:+2!Adam/batch_normalization_5/beta/m
2:002Adam/quant_conv2d/kernel/v
,:*02 Adam/batch_normalization/gamma/v
+:)02Adam/batch_normalization/beta/v
4:2002Adam/quant_conv2d_1/kernel/v
.:,02"Adam/batch_normalization_1/gamma/v
-:+02!Adam/batch_normalization_1/beta/v
4:2002Adam/quant_conv2d_2/kernel/v
.:,02"Adam/batch_normalization_2/gamma/v
-:+02!Adam/batch_normalization_2/beta/v
4:2002Adam/quant_conv2d_3/kernel/v
.:,02"Adam/batch_normalization_3/gamma/v
-:+02!Adam/batch_normalization_3/beta/v
+:)
АА2Adam/quant_dense/kernel/v
/:-А2"Adam/batch_normalization_4/gamma/v
.:,А2!Adam/batch_normalization_4/beta/v
,:*	А2Adam/quant_dense_1/kernel/v
.:,2"Adam/batch_normalization_5/gamma/v
-:+2!Adam/batch_normalization_5/beta/v
ж2г
F__inference_sequential_layer_call_and_return_conditional_losses_203412
F__inference_sequential_layer_call_and_return_conditional_losses_203639
F__inference_sequential_layer_call_and_return_conditional_losses_202503
F__inference_sequential_layer_call_and_return_conditional_losses_202585ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ъ2ч
+__inference_sequential_layer_call_fn_203704
+__inference_sequential_layer_call_fn_202733
+__inference_sequential_layer_call_fn_202880
+__inference_sequential_layer_call_fn_203769ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
у2р
!__inference__wrapped_model_200828 
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *:Ґ7
5К2
quant_conv2d_input€€€€€€€€€В
І2§
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_200857„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
М2Й
-__inference_quant_conv2d_layer_call_fn_200865„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
±2Ѓ
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_200871а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ц2У
.__inference_max_pooling2d_layer_call_fn_200877а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ю2ы
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203897
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203815
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203837
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203919і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
4__inference_batch_normalization_layer_call_fn_203863
4__inference_batch_normalization_layer_call_fn_203932
4__inference_batch_normalization_layer_call_fn_203945
4__inference_batch_normalization_layer_call_fn_203850і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©2¶
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_201067„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
О2Л
/__inference_quant_conv2d_1_layer_call_fn_201075„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
≥2∞
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_201081а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ш2Х
0__inference_max_pooling2d_1_layer_call_fn_201087а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ж2Г
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_204013
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_204073
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_204095
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_203991і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ъ2Ч
6__inference_batch_normalization_1_layer_call_fn_204121
6__inference_batch_normalization_1_layer_call_fn_204026
6__inference_batch_normalization_1_layer_call_fn_204108
6__inference_batch_normalization_1_layer_call_fn_204039і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©2¶
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_201277„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
О2Л
/__inference_quant_conv2d_2_layer_call_fn_201285„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
≥2∞
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_201291а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ш2Х
0__inference_max_pooling2d_2_layer_call_fn_201297а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ж2Г
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204189
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204271
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204167
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204249і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ъ2Ч
6__inference_batch_normalization_2_layer_call_fn_204215
6__inference_batch_normalization_2_layer_call_fn_204284
6__inference_batch_normalization_2_layer_call_fn_204202
6__inference_batch_normalization_2_layer_call_fn_204297і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©2¶
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_201487„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
О2Л
/__inference_quant_conv2d_3_layer_call_fn_201495„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
≥2∞
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_201501а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ш2Х
0__inference_max_pooling2d_3_layer_call_fn_201507а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ж2Г
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204343
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204365
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204425
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204447і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ъ2Ч
6__inference_batch_normalization_3_layer_call_fn_204460
6__inference_batch_normalization_3_layer_call_fn_204378
6__inference_batch_normalization_3_layer_call_fn_204473
6__inference_batch_normalization_3_layer_call_fn_204391і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
н2к
C__inference_flatten_layer_call_and_return_conditional_losses_204479Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_flatten_layer_call_fn_204484Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_quant_dense_layer_call_and_return_conditional_losses_204507Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_quant_dense_layer_call_fn_204514Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
а2Ё
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_204589
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_204612і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
™2І
6__inference_batch_normalization_4_layer_call_fn_204638
6__inference_batch_normalization_4_layer_call_fn_204625і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
у2р
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_204661Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_quant_dense_1_layer_call_fn_204668Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
а2Ё
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_204743
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_204766і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
™2І
6__inference_batch_normalization_5_layer_call_fn_204779
6__inference_batch_normalization_5_layer_call_fn_204792і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
р2н
F__inference_activation_layer_call_and_return_conditional_losses_204797Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_activation_layer_call_fn_204802Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
>B<
$__inference_signature_wrapper_203105quant_conv2d_input
о2л
D__inference_ste_sign_layer_call_and_return_conditional_losses_204814Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_ste_sign_layer_call_fn_204819Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_204831Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_ste_sign_1_layer_call_fn_204836Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_204848Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_ste_sign_2_layer_call_fn_204853Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_204865Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_ste_sign_3_layer_call_fn_204870Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_204882Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_ste_sign_4_layer_call_fn_204887Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_ste_sign_5_layer_call_and_return_conditional_losses_204899Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_ste_sign_5_layer_call_fn_204904Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_ste_sign_6_layer_call_and_return_conditional_losses_204916Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_ste_sign_6_layer_call_fn_204921Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 …
!__inference__wrapped_model_200828£"$%&'.89:;BLMNOV`abcnwtvu~ЗДЖЕDҐA
:Ґ7
5К2
quant_conv2d_input€€€€€€€€€В
™ "7™4
2

activation$К!

activation€€€€€€€€€Ґ
F__inference_activation_layer_call_and_return_conditional_losses_204797X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ z
+__inference_activation_layer_call_fn_204802K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€м
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_203991Ц89:;MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ м
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_204013Ц89:;MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ «
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_204073r89:;;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 0
p
™ "-Ґ*
#К 
0€€€€€€€€€ 0
Ъ «
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_204095r89:;;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 0
p 
™ "-Ґ*
#К 
0€€€€€€€€€ 0
Ъ ƒ
6__inference_batch_normalization_1_layer_call_fn_204026Й89:;MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0ƒ
6__inference_batch_normalization_1_layer_call_fn_204039Й89:;MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0Я
6__inference_batch_normalization_1_layer_call_fn_204108e89:;;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 0
p
™ " К€€€€€€€€€ 0Я
6__inference_batch_normalization_1_layer_call_fn_204121e89:;;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 0
p 
™ " К€€€€€€€€€ 0«
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204167rLMNO;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€0
p
™ "-Ґ*
#К 
0€€€€€€€€€0
Ъ «
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204189rLMNO;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€0
p 
™ "-Ґ*
#К 
0€€€€€€€€€0
Ъ м
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204249ЦLMNOMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ м
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_204271ЦLMNOMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ Я
6__inference_batch_normalization_2_layer_call_fn_204202eLMNO;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€0
p
™ " К€€€€€€€€€0Я
6__inference_batch_normalization_2_layer_call_fn_204215eLMNO;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€0
p 
™ " К€€€€€€€€€0ƒ
6__inference_batch_normalization_2_layer_call_fn_204284ЙLMNOMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0ƒ
6__inference_batch_normalization_2_layer_call_fn_204297ЙLMNOMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0м
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204343Ц`abcMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ м
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204365Ц`abcMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ «
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204425r`abc;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€0
p
™ "-Ґ*
#К 
0€€€€€€€€€0
Ъ «
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204447r`abc;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€0
p 
™ "-Ґ*
#К 
0€€€€€€€€€0
Ъ ƒ
6__inference_batch_normalization_3_layer_call_fn_204378Й`abcMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0ƒ
6__inference_batch_normalization_3_layer_call_fn_204391Й`abcMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0Я
6__inference_batch_normalization_3_layer_call_fn_204460e`abc;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€0
p
™ " К€€€€€€€€€0Я
6__inference_batch_normalization_3_layer_call_fn_204473e`abc;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€0
p 
™ " К€€€€€€€€€0є
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_204589dvwtu4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ є
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_204612dwtvu4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ С
6__inference_batch_normalization_4_layer_call_fn_204625Wvwtu4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€АС
6__inference_batch_normalization_4_layer_call_fn_204638Wwtvu4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€Аї
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_204743fЖЗДЕ3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ ї
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_204766fЗДЖЕ3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ У
6__inference_batch_normalization_5_layer_call_fn_204779YЖЗДЕ3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "К€€€€€€€€€У
6__inference_batch_normalization_5_layer_call_fn_204792YЗДЖЕ3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "К€€€€€€€€€к
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203815Ц$%&'MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ к
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203837Ц$%&'MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ ≈
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203897r$%&';Ґ8
1Ґ.
(К%
inputs€€€€€€€€€A
0
p
™ "-Ґ*
#К 
0€€€€€€€€€A
0
Ъ ≈
O__inference_batch_normalization_layer_call_and_return_conditional_losses_203919r$%&';Ґ8
1Ґ.
(К%
inputs€€€€€€€€€A
0
p 
™ "-Ґ*
#К 
0€€€€€€€€€A
0
Ъ ¬
4__inference_batch_normalization_layer_call_fn_203850Й$%&'MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0¬
4__inference_batch_normalization_layer_call_fn_203863Й$%&'MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0Э
4__inference_batch_normalization_layer_call_fn_203932e$%&';Ґ8
1Ґ.
(К%
inputs€€€€€€€€€A
0
p
™ " К€€€€€€€€€A
0Э
4__inference_batch_normalization_layer_call_fn_203945e$%&';Ґ8
1Ґ.
(К%
inputs€€€€€€€€€A
0
p 
™ " К€€€€€€€€€A
0®
C__inference_flatten_layer_call_and_return_conditional_losses_204479a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€0
™ "&Ґ#
К
0€€€€€€€€€А
Ъ А
(__inference_flatten_layer_call_fn_204484T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€0
™ "К€€€€€€€€€Ао
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_201081ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_1_layer_call_fn_201087СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_201291ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_2_layer_call_fn_201297СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_201501ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_3_layer_call_fn_201507СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€м
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_200871ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_max_pooling2d_layer_call_fn_200877СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ё
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_201067П.IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ ґ
/__inference_quant_conv2d_1_layer_call_fn_201075В.IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0ё
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_201277ПBIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ ґ
/__inference_quant_conv2d_2_layer_call_fn_201285ВBIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0ё
J__inference_quant_conv2d_3_layer_call_and_return_conditional_losses_201487ПVIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ ґ
/__inference_quant_conv2d_3_layer_call_fn_201495ВVIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0№
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_200857ПIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ і
-__inference_quant_conv2d_layer_call_fn_200865ВIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0©
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_204661\~0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ Б
.__inference_quant_dense_1_layer_call_fn_204668O~0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€®
G__inference_quant_dense_layer_call_and_return_conditional_losses_204507]n0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ А
,__inference_quant_dense_layer_call_fn_204514Pn0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€Ад
F__inference_sequential_layer_call_and_return_conditional_losses_202503Щ"$%&'.89:;BLMNOV`abcnvwtu~ЖЗДЕLҐI
BҐ?
5К2
quant_conv2d_input€€€€€€€€€В
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ д
F__inference_sequential_layer_call_and_return_conditional_losses_202585Щ"$%&'.89:;BLMNOV`abcnwtvu~ЗДЖЕLҐI
BҐ?
5К2
quant_conv2d_input€€€€€€€€€В
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ў
F__inference_sequential_layer_call_and_return_conditional_losses_203412Н"$%&'.89:;BLMNOV`abcnvwtu~ЖЗДЕ@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€В
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ў
F__inference_sequential_layer_call_and_return_conditional_losses_203639Н"$%&'.89:;BLMNOV`abcnwtvu~ЗДЖЕ@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€В
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Љ
+__inference_sequential_layer_call_fn_202733М"$%&'.89:;BLMNOV`abcnvwtu~ЖЗДЕLҐI
BҐ?
5К2
quant_conv2d_input€€€€€€€€€В
p

 
™ "К€€€€€€€€€Љ
+__inference_sequential_layer_call_fn_202880М"$%&'.89:;BLMNOV`abcnwtvu~ЗДЖЕLҐI
BҐ?
5К2
quant_conv2d_input€€€€€€€€€В
p 

 
™ "К€€€€€€€€€∞
+__inference_sequential_layer_call_fn_203704А"$%&'.89:;BLMNOV`abcnvwtu~ЖЗДЕ@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€В
p

 
™ "К€€€€€€€€€∞
+__inference_sequential_layer_call_fn_203769А"$%&'.89:;BLMNOV`abcnwtvu~ЗДЖЕ@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€В
p 

 
™ "К€€€€€€€€€в
$__inference_signature_wrapper_203105є"$%&'.89:;BLMNOV`abcnwtvu~ЗДЖЕZҐW
Ґ 
P™M
K
quant_conv2d_input5К2
quant_conv2d_input€€€€€€€€€В"7™4
2

activation$К!

activation€€€€€€€€€†
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_204831V.Ґ+
$Ґ!
К
inputs00
™ "$Ґ!
К
000
Ъ x
+__inference_ste_sign_1_layer_call_fn_204836I.Ґ+
$Ґ!
К
inputs00
™ "К00„
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_204848МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ Ѓ
+__inference_ste_sign_2_layer_call_fn_204853IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0†
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_204865V.Ґ+
$Ґ!
К
inputs00
™ "$Ґ!
К
000
Ъ x
+__inference_ste_sign_3_layer_call_fn_204870I.Ґ+
$Ґ!
К
inputs00
™ "К00„
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_204882МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ Ѓ
+__inference_ste_sign_4_layer_call_fn_204887IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0†
F__inference_ste_sign_5_layer_call_and_return_conditional_losses_204899V.Ґ+
$Ґ!
К
inputs00
™ "$Ґ!
К
000
Ъ x
+__inference_ste_sign_5_layer_call_fn_204904I.Ґ+
$Ґ!
К
inputs00
™ "К00„
F__inference_ste_sign_6_layer_call_and_return_conditional_losses_204916МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ Ѓ
+__inference_ste_sign_6_layer_call_fn_204921IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0Ю
D__inference_ste_sign_layer_call_and_return_conditional_losses_204814V.Ґ+
$Ґ!
К
inputs0
™ "$Ґ!
К
00
Ъ v
)__inference_ste_sign_layer_call_fn_204819I.Ґ+
$Ґ!
К
inputs0
™ "К0