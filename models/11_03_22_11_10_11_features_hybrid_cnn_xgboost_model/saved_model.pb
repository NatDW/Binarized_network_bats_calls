М√&
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
shapeshapeИ"serve*2.1.02v1.12.1-25073-g2c5e22190c8д» 
К
quant_conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*$
shared_namequant_conv2d/kernel
Г
'quant_conv2d/kernel/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel*&
_output_shapes
:8*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:8*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:8*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:8*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:8*
dtype0
О
quant_conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*&
shared_namequant_conv2d_1/kernel
З
)quant_conv2d_1/kernel/Read/ReadVariableOpReadVariableOpquant_conv2d_1/kernel*&
_output_shapes
:88*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:8*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:8*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:8*
dtype0
Ґ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:8*
dtype0
О
quant_conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*&
shared_namequant_conv2d_2/kernel
З
)quant_conv2d_2/kernel/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel*&
_output_shapes
:88*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:8*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:8*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:8*
dtype0
Ґ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:8*
dtype0
В
quant_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*#
shared_namequant_dense/kernel
{
&quant_dense/kernel/Read/ReadVariableOpReadVariableOpquant_dense/kernel* 
_output_shapes
:
АА*
dtype0
П
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_3/gamma
И
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_3/beta
Ж
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_3/moving_mean
Ф
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:А*
dtype0
£
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_3/moving_variance
Ь
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:А*
dtype0
Ж
quant_dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*%
shared_namequant_dense_1/kernel

(quant_dense_1/kernel/Read/ReadVariableOpReadVariableOpquant_dense_1/kernel* 
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
quant_dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*%
shared_namequant_dense_2/kernel
~
(quant_dense_2/kernel/Read/ReadVariableOpReadVariableOpquant_dense_2/kernel*
_output_shapes
:	А*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
Ґ
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance
Ы
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
Ш
Adam/quant_conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*+
shared_nameAdam/quant_conv2d/kernel/m
С
.Adam/quant_conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d/kernel/m*&
_output_shapes
:8*
dtype0
Ш
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*1
shared_name" Adam/batch_normalization/gamma/m
С
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:8*
dtype0
Ц
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*0
shared_name!Adam/batch_normalization/beta/m
П
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:8*
dtype0
Ь
Adam/quant_conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*-
shared_nameAdam/quant_conv2d_1/kernel/m
Х
0Adam/quant_conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_1/kernel/m*&
_output_shapes
:88*
dtype0
Ь
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"Adam/batch_normalization_1/gamma/m
Х
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:8*
dtype0
Ъ
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!Adam/batch_normalization_1/beta/m
У
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:8*
dtype0
Ь
Adam/quant_conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*-
shared_nameAdam/quant_conv2d_2/kernel/m
Х
0Adam/quant_conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_2/kernel/m*&
_output_shapes
:88*
dtype0
Ь
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"Adam/batch_normalization_2/gamma/m
Х
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:8*
dtype0
Ъ
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!Adam/batch_normalization_2/beta/m
У
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:8*
dtype0
Р
Adam/quant_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА**
shared_nameAdam/quant_dense/kernel/m
Й
-Adam/quant_dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_dense/kernel/m* 
_output_shapes
:
АА*
dtype0
Э
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_3/gamma/m
Ц
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_3/beta/m
Ф
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes	
:А*
dtype0
Ф
Adam/quant_dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*,
shared_nameAdam/quant_dense_1/kernel/m
Н
/Adam/quant_dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_dense_1/kernel/m* 
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
Adam/quant_dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*,
shared_nameAdam/quant_dense_2/kernel/m
М
/Adam/quant_dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/quant_dense_2/kernel/m*
_output_shapes
:	А*
dtype0
Ь
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/m
Х
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/m
У
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
:*
dtype0
Ш
Adam/quant_conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*+
shared_nameAdam/quant_conv2d/kernel/v
С
.Adam/quant_conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d/kernel/v*&
_output_shapes
:8*
dtype0
Ш
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*1
shared_name" Adam/batch_normalization/gamma/v
С
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:8*
dtype0
Ц
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*0
shared_name!Adam/batch_normalization/beta/v
П
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:8*
dtype0
Ь
Adam/quant_conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*-
shared_nameAdam/quant_conv2d_1/kernel/v
Х
0Adam/quant_conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_1/kernel/v*&
_output_shapes
:88*
dtype0
Ь
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"Adam/batch_normalization_1/gamma/v
Х
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:8*
dtype0
Ъ
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!Adam/batch_normalization_1/beta/v
У
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:8*
dtype0
Ь
Adam/quant_conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:88*-
shared_nameAdam/quant_conv2d_2/kernel/v
Х
0Adam/quant_conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_conv2d_2/kernel/v*&
_output_shapes
:88*
dtype0
Ь
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*3
shared_name$"Adam/batch_normalization_2/gamma/v
Х
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:8*
dtype0
Ъ
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:8*2
shared_name#!Adam/batch_normalization_2/beta/v
У
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:8*
dtype0
Р
Adam/quant_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА**
shared_nameAdam/quant_dense/kernel/v
Й
-Adam/quant_dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_dense/kernel/v* 
_output_shapes
:
АА*
dtype0
Э
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_3/gamma/v
Ц
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_3/beta/v
Ф
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes	
:А*
dtype0
Ф
Adam/quant_dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*,
shared_nameAdam/quant_dense_1/kernel/v
Н
/Adam/quant_dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_dense_1/kernel/v* 
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
Adam/quant_dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*,
shared_nameAdam/quant_dense_2/kernel/v
М
/Adam/quant_dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/quant_dense_2/kernel/v*
_output_shapes
:	А*
dtype0
Ь
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/v
Х
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/v
У
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
чЮ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*±Ю
value¶ЮBҐЮ BЪЮ
П
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
t
kernel_quantizer

kernel
	variables
regularization_losses
trainable_variables
	keras_api
Ч
axis
	gamma
 beta
!moving_mean
"moving_variance
#	variables
$regularization_losses
%trainable_variables
&	keras_api
R
'	variables
(regularization_losses
)trainable_variables
*	keras_api
Й
+kernel_quantizer
,input_quantizer

-kernel
.	variables
/regularization_losses
0trainable_variables
1	keras_api
Ч
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7	variables
8regularization_losses
9trainable_variables
:	keras_api
R
;	variables
<regularization_losses
=trainable_variables
>	keras_api
Й
?kernel_quantizer
@input_quantizer

Akernel
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
Ч
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
R
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
R
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
Й
Wkernel_quantizer
Xinput_quantizer

Ykernel
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
Ч
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
c	variables
dregularization_losses
etrainable_variables
f	keras_api
Й
gkernel_quantizer
hinput_quantizer

ikernel
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
Ч
naxis
	ogamma
pbeta
qmoving_mean
rmoving_variance
s	variables
tregularization_losses
utrainable_variables
v	keras_api
Й
wkernel_quantizer
xinput_quantizer

ykernel
z	variables
{regularization_losses
|trainable_variables
}	keras_api
Ю
~axis
	gamma
	Аbeta
Бmoving_mean
Вmoving_variance
Г	variables
Дregularization_losses
Еtrainable_variables
Ж	keras_api
V
З	variables
Иregularization_losses
Йtrainable_variables
К	keras_api
ѓ
	Лiter
Мbeta_1
Нbeta_2

Оdecay
Пlearning_ratemќmѕ m–-m—3m“4m”Am‘Gm’Hm÷Ym„_mЎ`mўimЏomџpm№ymЁmё	Аmяvаvб vв-vг3vд4vеAvжGvзHvиYvй_vк`vлivмovнpvоyvпvр	Аvс
й
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
А27
Б28
В29
 
З
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
А17
Ю
	variables
regularization_losses
Рnon_trainable_variables
 Сlayer_regularization_losses
Тmetrics
Уlayers
trainable_variables
 
l
Ф_custom_metrics
Х	variables
Цregularization_losses
Чtrainable_variables
Ш	keras_api
_]
VARIABLE_VALUEquant_conv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
Ю
	variables
regularization_losses
Щnon_trainable_variables
 Ъlayer_regularization_losses
Ыmetrics
Ьlayers
trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
 1
!2
"3
 

0
 1
Ю
#	variables
$regularization_losses
Эnon_trainable_variables
 Юlayer_regularization_losses
Яmetrics
†layers
%trainable_variables
 
 
 
Ю
'	variables
(regularization_losses
°non_trainable_variables
 Ґlayer_regularization_losses
£metrics
§layers
)trainable_variables
l
•_custom_metrics
¶	variables
Іregularization_losses
®trainable_variables
©	keras_api
V
™	variables
Ђregularization_losses
ђtrainable_variables
≠	keras_api
a_
VARIABLE_VALUEquant_conv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

-0
 

-0
Ю
.	variables
/regularization_losses
Ѓnon_trainable_variables
 ѓlayer_regularization_losses
∞metrics
±layers
0trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

30
41
52
63
 

30
41
Ю
7	variables
8regularization_losses
≤non_trainable_variables
 ≥layer_regularization_losses
іmetrics
µlayers
9trainable_variables
 
 
 
Ю
;	variables
<regularization_losses
ґnon_trainable_variables
 Јlayer_regularization_losses
Єmetrics
єlayers
=trainable_variables
l
Ї_custom_metrics
ї	variables
Љregularization_losses
љtrainable_variables
Њ	keras_api
V
њ	variables
јregularization_losses
Ѕtrainable_variables
¬	keras_api
a_
VARIABLE_VALUEquant_conv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

A0
 

A0
Ю
B	variables
Cregularization_losses
√non_trainable_variables
 ƒlayer_regularization_losses
≈metrics
∆layers
Dtrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
I2
J3
 

G0
H1
Ю
K	variables
Lregularization_losses
«non_trainable_variables
 »layer_regularization_losses
…metrics
 layers
Mtrainable_variables
 
 
 
Ю
O	variables
Pregularization_losses
Ћnon_trainable_variables
 ћlayer_regularization_losses
Ќmetrics
ќlayers
Qtrainable_variables
 
 
 
Ю
S	variables
Tregularization_losses
ѕnon_trainable_variables
 –layer_regularization_losses
—metrics
“layers
Utrainable_variables
l
”_custom_metrics
‘	variables
’regularization_losses
÷trainable_variables
„	keras_api
V
Ў	variables
ўregularization_losses
Џtrainable_variables
џ	keras_api
^\
VARIABLE_VALUEquant_dense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE

Y0
 

Y0
Ю
Z	variables
[regularization_losses
№non_trainable_variables
 Ёlayer_regularization_losses
ёmetrics
яlayers
\trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

_0
`1
a2
b3
 

_0
`1
Ю
c	variables
dregularization_losses
аnon_trainable_variables
 бlayer_regularization_losses
вmetrics
гlayers
etrainable_variables
l
д_custom_metrics
е	variables
жregularization_losses
зtrainable_variables
и	keras_api
V
й	variables
кregularization_losses
лtrainable_variables
м	keras_api
`^
VARIABLE_VALUEquant_dense_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE

i0
 

i0
Ю
j	variables
kregularization_losses
нnon_trainable_variables
 оlayer_regularization_losses
пmetrics
рlayers
ltrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

o0
p1
q2
r3
 

o0
p1
Ю
s	variables
tregularization_losses
сnon_trainable_variables
 тlayer_regularization_losses
уmetrics
фlayers
utrainable_variables
l
х_custom_metrics
ц	variables
чregularization_losses
шtrainable_variables
щ	keras_api
V
ъ	variables
ыregularization_losses
ьtrainable_variables
э	keras_api
a_
VARIABLE_VALUEquant_dense_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE

y0
 

y0
Ю
z	variables
{regularization_losses
юnon_trainable_variables
 €layer_regularization_losses
Аmetrics
Бlayers
|trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
А1
Б2
В3
 

0
А1
°
Г	variables
Дregularization_losses
Вnon_trainable_variables
 Гlayer_regularization_losses
Дmetrics
Еlayers
Еtrainable_variables
 
 
 
°
З	variables
Иregularization_losses
Жnon_trainable_variables
 Зlayer_regularization_losses
Иmetrics
Йlayers
Йtrainable_variables
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
Б10
В11
 

К0
Л1
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
°
Х	variables
Цregularization_losses
Мnon_trainable_variables
 Нlayer_regularization_losses
Оmetrics
Пlayers
Чtrainable_variables
 
 
 

0
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
 
 
°
¶	variables
Іregularization_losses
Рnon_trainable_variables
 Сlayer_regularization_losses
Тmetrics
Уlayers
®trainable_variables
 
 
 
°
™	variables
Ђregularization_losses
Фnon_trainable_variables
 Хlayer_regularization_losses
Цmetrics
Чlayers
ђtrainable_variables
 
 
 

+0
,1
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
 
 
°
ї	variables
Љregularization_losses
Шnon_trainable_variables
 Щlayer_regularization_losses
Ъmetrics
Ыlayers
љtrainable_variables
 
 
 
°
њ	variables
јregularization_losses
Ьnon_trainable_variables
 Эlayer_regularization_losses
Юmetrics
Яlayers
Ѕtrainable_variables
 
 
 

?0
@1
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
 
 
°
‘	variables
’regularization_losses
†non_trainable_variables
 °layer_regularization_losses
Ґmetrics
£layers
÷trainable_variables
 
 
 
°
Ў	variables
ўregularization_losses
§non_trainable_variables
 •layer_regularization_losses
¶metrics
Іlayers
Џtrainable_variables
 
 
 

W0
X1

a0
b1
 
 
 
 
 
 
 
°
е	variables
жregularization_losses
®non_trainable_variables
 ©layer_regularization_losses
™metrics
Ђlayers
зtrainable_variables
 
 
 
°
й	variables
кregularization_losses
ђnon_trainable_variables
 ≠layer_regularization_losses
Ѓmetrics
ѓlayers
лtrainable_variables
 
 
 

g0
h1

q0
r1
 
 
 
 
 
 
 
°
ц	variables
чregularization_losses
∞non_trainable_variables
 ±layer_regularization_losses
≤metrics
≥layers
шtrainable_variables
 
 
 
°
ъ	variables
ыregularization_losses
іnon_trainable_variables
 µlayer_regularization_losses
ґmetrics
Јlayers
ьtrainable_variables
 
 
 

w0
x1

Б0
В1
 
 
 
 
 
 
 


Єtotal

єcount
Ї
_fn_kwargs
ї	variables
Љregularization_losses
љtrainable_variables
Њ	keras_api


њtotal

јcount
Ѕ
_fn_kwargs
¬	variables
√regularization_losses
ƒtrainable_variables
≈	keras_api
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
Є0
є1
 
 
°
ї	variables
Љregularization_losses
∆non_trainable_variables
 «layer_regularization_losses
»metrics
…layers
љtrainable_variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

њ0
ј1
 
 
°
¬	variables
√regularization_losses
 non_trainable_variables
 Ћlayer_regularization_losses
ћmetrics
Ќlayers
ƒtrainable_variables

Є0
є1
 
 
 

њ0
ј1
 
 
 
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
Б
VARIABLE_VALUEAdam/quant_dense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEAdam/quant_dense_1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/quant_dense_2/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
Б
VARIABLE_VALUEAdam/quant_dense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEAdam/quant_dense_1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/quant_dense_2/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
•	
StatefulPartitionedCallStatefulPartitionedCall"serving_default_quant_conv2d_inputquant_conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancequant_conv2d_1/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancequant_conv2d_2/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancequant_dense/kernel%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betaquant_dense_1/kernel%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betaquant_dense_2/kernel%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/beta**
Tin#
!2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_171733
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Щ
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
__inference__traced_save_173746
Є
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
"__inference__traced_restore_173983Ш√
≠%
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172881

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_172866
assignmovingavg_1_172873
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 8:8:8:8:8:*
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
loc:@AssignMovingAvg/172866*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/172866*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_172866*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/172866*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/172866*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_172866AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/172866*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/172873*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172873*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_172873*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172873*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172873*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_172873AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/172873*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€ 8
 
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
у%
Ъ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_169679

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_169664
assignmovingavg_1_169671
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
loc:@AssignMovingAvg/169664*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/169664*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_169664*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/169664*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/169664*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_169664AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/169664*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/169671*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/169671*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_169671*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/169671*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/169671*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_169671AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/169671*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
+__inference_sequential_layer_call_fn_171368
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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1713052
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

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
м
L
0__inference_max_pooling2d_1_layer_call_fn_169947

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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1699412
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
†
r
,__inference_quant_dense_layer_call_fn_172970

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
G__inference_quant_dense_layer_call_and_return_conditional_losses_1709392
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€А:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: 
М
Ґ
G__inference_quant_dense_layer_call_and_return_conditional_losses_172963

inputs"
matmul_readvariableop_resource
identityИҐMatMul/ReadVariableOpe
ste_sign_6/SignSigninputs*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_6/Signi
ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
ste_sign_6/add/yМ
ste_sign_6/addAddV2ste_sign_6/Sign:y:0ste_sign_6/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_6/addu
ste_sign_6/Sign_1Signste_sign_6/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_6/Sign_1А
ste_sign_6/IdentityIdentityste_sign_6/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_6/Identity—
ste_sign_6/IdentityN	IdentityNste_sign_6/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-172943*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2
ste_sign_6/IdentityNП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpВ
MatMul/ste_sign_5/SignSignMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_5/Signw
MatMul/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
MatMul/ste_sign_5/add/y†
MatMul/ste_sign_5/addAddV2MatMul/ste_sign_5/Sign:y:0 MatMul/ste_sign_5/add/y:output:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_5/addВ
MatMul/ste_sign_5/Sign_1SignMatMul/ste_sign_5/add:z:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_5/Sign_1Н
MatMul/ste_sign_5/IdentityIdentityMatMul/ste_sign_5/Sign_1:y:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_5/Identityн
MatMul/ste_sign_5/IdentityN	IdentityNMatMul/ste_sign_5/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-172953*,
_output_shapes
:
АА:
АА2
MatMul/ste_sign_5/IdentityNТ
MatMulMatMulste_sign_6/IdentityN:output:0$MatMul/ste_sign_5/IdentityN:output:0*
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
:€€€€€€€€€А:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: 
И
©
6__inference_batch_normalization_2_layer_call_fn_172929

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
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1708652
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 8
 
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
+__inference_ste_sign_1_layer_call_fn_173446

inputs
identityВ
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
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_1697762
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
€
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_169731

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
у%
Ъ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172447

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_172432
assignmovingavg_1_172439
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
loc:@AssignMovingAvg/172432*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/172432*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_172432*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/172432*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/172432*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_172432AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/172432*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/172439*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172439*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_172439*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172439*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172439*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_172439AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/172439*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
F__inference_activation_layer_call_and_return_conditional_losses_173407

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ЏЯ
ѓ$
__inference__traced_save_173746
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

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1•
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1b75e61d612640afb7c4cf95eb89661a/part2
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
SaveV2/shape_and_slicesч"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_quant_conv2d_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop0savev2_quant_conv2d_1_kernel_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop0savev2_quant_conv2d_2_kernel_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop-savev2_quant_dense_kernel_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop/savev2_quant_dense_1_kernel_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop/savev2_quant_dense_2_kernel_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_quant_conv2d_kernel_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop7savev2_adam_quant_conv2d_1_kernel_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop7savev2_adam_quant_conv2d_2_kernel_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop4savev2_adam_quant_dense_kernel_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop6savev2_adam_quant_dense_1_kernel_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop6savev2_adam_quant_dense_2_kernel_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop5savev2_adam_quant_conv2d_kernel_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop7savev2_adam_quant_conv2d_1_kernel_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop7savev2_adam_quant_conv2d_2_kernel_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop4savev2_adam_quant_dense_kernel_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop6savev2_adam_quant_dense_1_kernel_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop6savev2_adam_quant_dense_2_kernel_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop"/device:CPU:0*
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

identity_1Identity_1:output:0*ж
_input_shapes‘
—: :8:8:8:8:8:88:8:8:8:8:88:8:8:8:8:
АА:А:А:А:А:
АА:А:А:А:А:	А::::: : : : : : : : : :8:8:8:88:8:8:88:8:8:
АА:А:А:
АА:А:А:	А:::8:8:8:88:8:8:88:8:8:
АА:А:А:
АА:А:А:	А::: 2(
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
АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:&"
 
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
:	А: 
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
АА:!2

_output_shapes	
:А:!3

_output_shapes	
:А:&4"
 
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
:	А: 8
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
АА:!D

_output_shapes	
:А:!E

_output_shapes	
:А:&F"
 
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
:	А: J

_output_shapes
:: K

_output_shapes
::L

_output_shapes
: 
њ
Ј
+__inference_sequential_layer_call_fn_172401

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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1714512
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

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
К
u
/__inference_quant_conv2d_1_layer_call_fn_169795

inputs
unknown
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1697872
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
_user_specified_nameinputs:

_output_shapes
: 
ъ
И
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_170602

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
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/add_1џ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
+__inference_sequential_layer_call_fn_171514
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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1714512
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

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
≈	
d
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_169963

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identityв
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-169954*n
_output_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
	IdentityNД

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
_user_specified_nameinputs
ѓ%
Ъ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172529

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_172514
assignmovingavg_1_172521
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
Const_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*L
_output_shapes:
8:€€€€€€€€€В8:8:8:8:8:*
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
loc:@AssignMovingAvg/172514*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/172514*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_172514*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/172514*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/172514*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_172514AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/172514*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/172521*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172521*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_172521*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172521*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172521*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_172521AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/172521*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:€€€€€€€€€В82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€В8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€В8
 
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
Ј
Љ
$__inference_signature_wrapper_171733
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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_1695482
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

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
’0
»
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_173199

inputs
assignmovingavg_173174
assignmovingavg_1_173180)
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
loc:@AssignMovingAvg/173174*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_173174*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpƒ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/173174*
_output_shapes	
:А2
AssignMovingAvg/subї
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/173174*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_173174AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/173174*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/173180*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_173180*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpќ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173180*
_output_shapes	
:А2
AssignMovingAvg_1/sub≈
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173180*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_173180AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/173180*
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
“Y
µ
F__inference_sequential_layer_call_and_return_conditional_losses_171451

inputs
quant_conv2d_171373
batch_normalization_171376
batch_normalization_171378
batch_normalization_171380
batch_normalization_171382
quant_conv2d_1_171386 
batch_normalization_1_171389 
batch_normalization_1_171391 
batch_normalization_1_171393 
batch_normalization_1_171395
quant_conv2d_2_171399 
batch_normalization_2_171402 
batch_normalization_2_171404 
batch_normalization_2_171406 
batch_normalization_2_171408
quant_dense_171413 
batch_normalization_3_171416 
batch_normalization_3_171418 
batch_normalization_3_171420 
batch_normalization_3_171422
quant_dense_1_171425 
batch_normalization_4_171428 
batch_normalization_4_171430 
batch_normalization_4_171432 
batch_normalization_4_171434
quant_dense_2_171437 
batch_normalization_5_171440 
batch_normalization_5_171442 
batch_normalization_5_171444 
batch_normalization_5_171446
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ$quant_conv2d/StatefulPartitionedCallҐ&quant_conv2d_1/StatefulPartitionedCallҐ&quant_conv2d_2/StatefulPartitionedCallҐ#quant_dense/StatefulPartitionedCallҐ%quant_dense_1/StatefulPartitionedCallҐ%quant_dense_2/StatefulPartitionedCall÷
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsquant_conv2d_171373*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В8*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1695772&
$quant_conv2d/StatefulPartitionedCallр
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_171376batch_normalization_171378batch_normalization_171380batch_normalization_171382*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1706752-
+batch_normalization/StatefulPartitionedCallЎ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1697312
max_pooling2d/PartitionedCallэ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_171386*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1697872(
&quant_conv2d_1/StatefulPartitionedCall€
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_171389batch_normalization_1_171391batch_normalization_1_171393batch_normalization_1_171395*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1707702/
-batch_normalization_1/StatefulPartitionedCallа
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1699412!
max_pooling2d_1/PartitionedCall€
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_171399*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1699972(
&quant_conv2d_2/StatefulPartitionedCall€
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_171402batch_normalization_2_171404batch_normalization_2_171406batch_normalization_2_171408*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1708652/
-batch_normalization_2/StatefulPartitionedCallа
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1701512!
max_pooling2d_2/PartitionedCall≥
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1709082
flatten/PartitionedCallд
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_171413*
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
G__inference_quant_dense_layer_call_and_return_conditional_losses_1709392%
#quant_dense/StatefulPartitionedCallх
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_171416batch_normalization_3_171418batch_normalization_3_171420batch_normalization_3_171422*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1702982/
-batch_normalization_3/StatefulPartitionedCallВ
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_171425*
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
GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1710092'
%quant_dense_1/StatefulPartitionedCallч
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_171428batch_normalization_4_171430batch_normalization_4_171432batch_normalization_4_171434*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1704502/
-batch_normalization_4/StatefulPartitionedCallБ
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_171437*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1710792'
%quant_dense_2/StatefulPartitionedCallц
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_171440batch_normalization_5_171442batch_normalization_5_171444batch_normalization_5_171446*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1706022/
-batch_normalization_5/StatefulPartitionedCall…
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1711312
activation/PartitionedCallД
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

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
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall:X T
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
≈	
d
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_173492

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identityв
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-173483*n
_output_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
	IdentityNД

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
_user_specified_nameinputs
ф
ф
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_170770

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
8:8:8:8:8:*
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
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€A
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
–
©
6__inference_batch_normalization_1_layer_call_fn_172658

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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1698892
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_170262

inputs
assignmovingavg_170237
assignmovingavg_1_170243)
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
loc:@AssignMovingAvg/170237*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_170237*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpƒ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/170237*
_output_shapes	
:А2
AssignMovingAvg/subї
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/170237*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_170237AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/170237*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/170243*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_170243*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpќ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170243*
_output_shapes	
:А2
AssignMovingAvg_1/sub≈
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170243*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_170243AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/170243*
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
К
u
/__inference_quant_conv2d_2_layer_call_fn_170005

inputs
unknown
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1699972
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
_user_specified_nameinputs:

_output_shapes
: 
ї
_
C__inference_flatten_layer_call_and_return_conditional_losses_172935

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€8:W S
/
_output_shapes
:€€€€€€€€€8
 
_user_specified_nameinputs
И
£
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_169577

inputs"
conv2d_readvariableop_resource
identityИҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02
Conv2D/ReadVariableOpЈ
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
D__inference_ste_sign_layer_call_and_return_conditional_losses_1695662!
Conv2D/ste_sign/PartitionedCallЫ
Conv2D/ste_sign/IdentityIdentity(Conv2D/ste_sign/PartitionedCall:output:0*
T0*&
_output_shapes
:82
Conv2D/ste_sign/Identityє
Conv2DConv2Dinputs!Conv2D/ste_sign/Identity:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*
paddingSAME*
strides
2
Conv2DХ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

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
Љ
b
F__inference_activation_layer_call_and_return_conditional_losses_171131

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
љ0
»
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_170566

inputs
assignmovingavg_170541
assignmovingavg_1_170547)
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
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

:*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/170541*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_170541*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp√
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/170541*
_output_shapes
:2
AssignMovingAvg/subЇ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/170541*
_output_shapes
:2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_170541AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/170541*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/170547*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_170547*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЌ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170547*
_output_shapes
:2
AssignMovingAvg_1/subƒ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170547*
_output_shapes
:2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_170547AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/170547*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/add_1≥
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
D__inference_ste_sign_layer_call_and_return_conditional_losses_169566

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
 *Ќћћ=2
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

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-169557*8
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
љ0
»
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_173353

inputs
assignmovingavg_173328
assignmovingavg_1_173334)
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient§
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
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

:*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Ю
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/173328*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_173328*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp√
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/173328*
_output_shapes
:2
AssignMovingAvg/subЇ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/173328*
_output_shapes
:2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_173328AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/173328*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/173334*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_173334*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЌ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173334*
_output_shapes
:2
AssignMovingAvg_1/subƒ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173334*
_output_shapes
:2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_173334AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/173334*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/add_1≥
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
6__inference_batch_normalization_4_layer_call_fn_173248

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1704502
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
ъ
И
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_173376

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
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
batchnorm/add_1џ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
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
ос
“
!__inference__wrapped_model_169548
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
identityИҐ>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpҐ@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ-sequential/batch_normalization/ReadVariableOpҐ/sequential/batch_normalization/ReadVariableOp_1Ґ@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐBsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ/sequential/batch_normalization_1/ReadVariableOpҐ1sequential/batch_normalization_1/ReadVariableOp_1Ґ@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐBsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ/sequential/batch_normalization_2/ReadVariableOpҐ1sequential/batch_normalization_2/ReadVariableOp_1Ґ9sequential/batch_normalization_3/batchnorm/ReadVariableOpҐ;sequential/batch_normalization_3/batchnorm/ReadVariableOp_1Ґ;sequential/batch_normalization_3/batchnorm/ReadVariableOp_2Ґ=sequential/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ9sequential/batch_normalization_4/batchnorm/ReadVariableOpҐ;sequential/batch_normalization_4/batchnorm/ReadVariableOp_1Ґ;sequential/batch_normalization_4/batchnorm/ReadVariableOp_2Ґ=sequential/batch_normalization_4/batchnorm/mul/ReadVariableOpҐ9sequential/batch_normalization_5/batchnorm/ReadVariableOpҐ;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1Ґ;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2Ґ=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOpҐ-sequential/quant_conv2d/Conv2D/ReadVariableOpҐ/sequential/quant_conv2d_1/Conv2D/ReadVariableOpҐ/sequential/quant_conv2d_2/Conv2D/ReadVariableOpҐ,sequential/quant_dense/MatMul/ReadVariableOpҐ.sequential/quant_dense_1/MatMul/ReadVariableOpҐ.sequential/quant_dense_2/MatMul/ReadVariableOpЁ
-sequential/quant_conv2d/Conv2D/ReadVariableOpReadVariableOp6sequential_quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02/
-sequential/quant_conv2d/Conv2D/ReadVariableOpћ
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
 *Ќћћ=2/
-sequential/quant_conv2d/Conv2D/ste_sign/add/yю
+sequential/quant_conv2d/Conv2D/ste_sign/addAddV20sequential/quant_conv2d/Conv2D/ste_sign/Sign:y:06sequential/quant_conv2d/Conv2D/ste_sign/add/y:output:0*
T0*&
_output_shapes
:82-
+sequential/quant_conv2d/Conv2D/ste_sign/add 
.sequential/quant_conv2d/Conv2D/ste_sign/Sign_1Sign/sequential/quant_conv2d/Conv2D/ste_sign/add:z:0*
T0*&
_output_shapes
:820
.sequential/quant_conv2d/Conv2D/ste_sign/Sign_1’
0sequential/quant_conv2d/Conv2D/ste_sign/IdentityIdentity2sequential/quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*
T0*&
_output_shapes
:822
0sequential/quant_conv2d/Conv2D/ste_sign/Identity”
1sequential/quant_conv2d/Conv2D/ste_sign/IdentityN	IdentityN2sequential/quant_conv2d/Conv2D/ste_sign/Sign_1:y:05sequential/quant_conv2d/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-169326*8
_output_shapes&
$:8:823
1sequential/quant_conv2d/Conv2D/ste_sign/IdentityNэ
sequential/quant_conv2d/Conv2DConv2Dquant_conv2d_input:sequential/quant_conv2d/Conv2D/ste_sign/IdentityN:output:0*
T0*0
_output_shapes
:€€€€€€€€€В8*
paddingSAME*
strides
2 
sequential/quant_conv2d/Conv2DЬ
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
:8*
dtype02/
-sequential/batch_normalization/ReadVariableOp„
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype021
/sequential/batch_normalization/ReadVariableOp_1Д
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpК
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¶
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3'sequential/quant_conv2d/Conv2D:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:€€€€€€€€€В8:8:8:8:8:*
epsilon%oГ:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3С
$sequential/batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2&
$sequential/batch_normalization/Constс
 sequential/max_pooling2d/MaxPoolMaxPool3sequential/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€A
8*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool√
)sequential/quant_conv2d_1/ste_sign_2/SignSign)sequential/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
82+
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
82*
(sequential/quant_conv2d_1/ste_sign_2/add 
+sequential/quant_conv2d_1/ste_sign_2/Sign_1Sign,sequential/quant_conv2d_1/ste_sign_2/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€A
82-
+sequential/quant_conv2d_1/ste_sign_2/Sign_1’
-sequential/quant_conv2d_1/ste_sign_2/IdentityIdentity/sequential/quant_conv2d_1/ste_sign_2/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€A
82/
-sequential/quant_conv2d_1/ste_sign_2/Identity–
.sequential/quant_conv2d_1/ste_sign_2/IdentityN	IdentityN/sequential/quant_conv2d_1/ste_sign_2/Sign_1:y:0)sequential/max_pooling2d/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-169354*J
_output_shapes8
6:€€€€€€€€€A
8:€€€€€€€€€A
820
.sequential/quant_conv2d_1/ste_sign_2/IdentityNг
/sequential/quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp8sequential_quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype021
/sequential/quant_conv2d_1/Conv2D/ReadVariableOp÷
0sequential/quant_conv2d_1/Conv2D/ste_sign_1/SignSign7sequential/quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:8822
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
:8821
/sequential/quant_conv2d_1/Conv2D/ste_sign_1/add÷
2sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1Sign3sequential/quant_conv2d_1/Conv2D/ste_sign_1/add:z:0*
T0*&
_output_shapes
:8824
2sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1б
4sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityIdentity6sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0*
T0*&
_output_shapes
:8826
4sequential/quant_conv2d_1/Conv2D/ste_sign_1/Identityб
5sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityN	IdentityN6sequential/quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:07sequential/quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-169364*8
_output_shapes&
$:88:8827
5sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityN©
 sequential/quant_conv2d_1/Conv2DConv2D7sequential/quant_conv2d_1/ste_sign_2/IdentityN:output:0>sequential/quant_conv2d_1/Conv2D/ste_sign_1/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
8*
paddingSAME*
strides
2"
 sequential/quant_conv2d_1/Conv2D†
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
:8*
dtype021
/sequential/batch_normalization_1/ReadVariableOpЁ
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:8*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1К
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpР
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1≥
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3)sequential/quant_conv2d_1/Conv2D:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
8:8:8:8:8:*
epsilon%oГ:*
is_training( 23
1sequential/batch_normalization_1/FusedBatchNormV3Х
&sequential/batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2(
&sequential/batch_normalization_1/Constч
"sequential/max_pooling2d_1/MaxPoolMaxPool5sequential/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€ 8*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool≈
)sequential/quant_conv2d_2/ste_sign_4/SignSign+sequential/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 82+
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
:€€€€€€€€€ 82*
(sequential/quant_conv2d_2/ste_sign_4/add 
+sequential/quant_conv2d_2/ste_sign_4/Sign_1Sign,sequential/quant_conv2d_2/ste_sign_4/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€ 82-
+sequential/quant_conv2d_2/ste_sign_4/Sign_1’
-sequential/quant_conv2d_2/ste_sign_4/IdentityIdentity/sequential/quant_conv2d_2/ste_sign_4/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 82/
-sequential/quant_conv2d_2/ste_sign_4/Identity“
.sequential/quant_conv2d_2/ste_sign_4/IdentityN	IdentityN/sequential/quant_conv2d_2/ste_sign_4/Sign_1:y:0+sequential/max_pooling2d_1/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-169392*J
_output_shapes8
6:€€€€€€€€€ 8:€€€€€€€€€ 820
.sequential/quant_conv2d_2/ste_sign_4/IdentityNг
/sequential/quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp8sequential_quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype021
/sequential/quant_conv2d_2/Conv2D/ReadVariableOp÷
0sequential/quant_conv2d_2/Conv2D/ste_sign_3/SignSign7sequential/quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:8822
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
:8821
/sequential/quant_conv2d_2/Conv2D/ste_sign_3/add÷
2sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1Sign3sequential/quant_conv2d_2/Conv2D/ste_sign_3/add:z:0*
T0*&
_output_shapes
:8824
2sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1б
4sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityIdentity6sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0*
T0*&
_output_shapes
:8826
4sequential/quant_conv2d_2/Conv2D/ste_sign_3/Identityб
5sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityN	IdentityN6sequential/quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:07sequential/quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-169402*8
_output_shapes&
$:88:8827
5sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityN©
 sequential/quant_conv2d_2/Conv2DConv2D7sequential/quant_conv2d_2/ste_sign_4/IdentityN:output:0>sequential/quant_conv2d_2/Conv2D/ste_sign_3/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 8*
paddingSAME*
strides
2"
 sequential/quant_conv2d_2/Conv2D†
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
:8*
dtype021
/sequential/batch_normalization_2/ReadVariableOpЁ
1sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:8*
dtype023
1sequential/batch_normalization_2/ReadVariableOp_1К
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02B
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpР
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02D
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1≥
1sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3)sequential/quant_conv2d_2/Conv2D:output:07sequential/batch_normalization_2/ReadVariableOp:value:09sequential/batch_normalization_2/ReadVariableOp_1:value:0Hsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 8:8:8:8:8:*
epsilon%oГ:*
is_training( 23
1sequential/batch_normalization_2/FusedBatchNormV3Х
&sequential/batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2(
&sequential/batch_normalization_2/Constч
"sequential/max_pooling2d_2/MaxPoolMaxPool5sequential/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€8*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPoolЕ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
sequential/flatten/Const∆
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_2/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/flatten/Reshape∞
&sequential/quant_dense/ste_sign_6/SignSign#sequential/flatten/Reshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential/quant_dense/ste_sign_6/SignЧ
'sequential/quant_dense/ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2)
'sequential/quant_dense/ste_sign_6/add/yи
%sequential/quant_dense/ste_sign_6/addAddV2*sequential/quant_dense/ste_sign_6/Sign:y:00sequential/quant_dense/ste_sign_6/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%sequential/quant_dense/ste_sign_6/addЇ
(sequential/quant_dense/ste_sign_6/Sign_1Sign)sequential/quant_dense/ste_sign_6/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(sequential/quant_dense/ste_sign_6/Sign_1≈
*sequential/quant_dense/ste_sign_6/IdentityIdentity,sequential/quant_dense/ste_sign_6/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2,
*sequential/quant_dense/ste_sign_6/Identity≥
+sequential/quant_dense/ste_sign_6/IdentityN	IdentityN,sequential/quant_dense/ste_sign_6/Sign_1:y:0#sequential/flatten/Reshape:output:0*
T
2*,
_gradient_op_typeCustomGradient-169432*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2-
+sequential/quant_dense/ste_sign_6/IdentityN‘
,sequential/quant_dense/MatMul/ReadVariableOpReadVariableOp5sequential_quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02.
,sequential/quant_dense/MatMul/ReadVariableOp«
-sequential/quant_dense/MatMul/ste_sign_5/SignSign4sequential/quant_dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2/
-sequential/quant_dense/MatMul/ste_sign_5/Sign•
.sequential/quant_dense/MatMul/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=20
.sequential/quant_dense/MatMul/ste_sign_5/add/yь
,sequential/quant_dense/MatMul/ste_sign_5/addAddV21sequential/quant_dense/MatMul/ste_sign_5/Sign:y:07sequential/quant_dense/MatMul/ste_sign_5/add/y:output:0*
T0* 
_output_shapes
:
АА2.
,sequential/quant_dense/MatMul/ste_sign_5/add«
/sequential/quant_dense/MatMul/ste_sign_5/Sign_1Sign0sequential/quant_dense/MatMul/ste_sign_5/add:z:0*
T0* 
_output_shapes
:
АА21
/sequential/quant_dense/MatMul/ste_sign_5/Sign_1“
1sequential/quant_dense/MatMul/ste_sign_5/IdentityIdentity3sequential/quant_dense/MatMul/ste_sign_5/Sign_1:y:0*
T0* 
_output_shapes
:
АА23
1sequential/quant_dense/MatMul/ste_sign_5/Identity…
2sequential/quant_dense/MatMul/ste_sign_5/IdentityN	IdentityN3sequential/quant_dense/MatMul/ste_sign_5/Sign_1:y:04sequential/quant_dense/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-169442*,
_output_shapes
:
АА:
АА24
2sequential/quant_dense/MatMul/ste_sign_5/IdentityNо
sequential/quant_dense/MatMulMatMul4sequential/quant_dense/ste_sign_6/IdentityN:output:0;sequential/quant_dense/MatMul/ste_sign_5/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/quant_dense/MatMul†
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
+sequential/batch_normalization_3/LogicalAndц
9sequential/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpBsequential_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9sequential/batch_normalization_3/batchnorm/ReadVariableOp©
0sequential/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:22
0sequential/batch_normalization_3/batchnorm/add/yН
.sequential/batch_normalization_3/batchnorm/addAddV2Asequential/batch_normalization_3/batchnorm/ReadVariableOp:value:09sequential/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А20
.sequential/batch_normalization_3/batchnorm/add«
0sequential/batch_normalization_3/batchnorm/RsqrtRsqrt2sequential/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:А22
0sequential/batch_normalization_3/batchnorm/RsqrtВ
=sequential/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpFsequential_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02?
=sequential/batch_normalization_3/batchnorm/mul/ReadVariableOpК
.sequential/batch_normalization_3/batchnorm/mulMul4sequential/batch_normalization_3/batchnorm/Rsqrt:y:0Esequential/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А20
.sequential/batch_normalization_3/batchnorm/mulы
0sequential/batch_normalization_3/batchnorm/mul_1Mul'sequential/quant_dense/MatMul:product:02sequential/batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А22
0sequential/batch_normalization_3/batchnorm/mul_1ь
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02=
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_1К
0sequential/batch_normalization_3/batchnorm/mul_2MulCsequential/batch_normalization_3/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:А22
0sequential/batch_normalization_3/batchnorm/mul_2ь
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpDsequential_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02=
;sequential/batch_normalization_3/batchnorm/ReadVariableOp_2И
.sequential/batch_normalization_3/batchnorm/subSubCsequential/batch_normalization_3/batchnorm/ReadVariableOp_2:value:04sequential/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А20
.sequential/batch_normalization_3/batchnorm/subК
0sequential/batch_normalization_3/batchnorm/add_1AddV24sequential/batch_normalization_3/batchnorm/mul_1:z:02sequential/batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А22
0sequential/batch_normalization_3/batchnorm/add_1≈
(sequential/quant_dense_1/ste_sign_8/SignSign4sequential/batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(sequential/quant_dense_1/ste_sign_8/SignЫ
)sequential/quant_dense_1/ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2+
)sequential/quant_dense_1/ste_sign_8/add/yр
'sequential/quant_dense_1/ste_sign_8/addAddV2,sequential/quant_dense_1/ste_sign_8/Sign:y:02sequential/quant_dense_1/ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2)
'sequential/quant_dense_1/ste_sign_8/addј
*sequential/quant_dense_1/ste_sign_8/Sign_1Sign+sequential/quant_dense_1/ste_sign_8/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2,
*sequential/quant_dense_1/ste_sign_8/Sign_1Ћ
,sequential/quant_dense_1/ste_sign_8/IdentityIdentity.sequential/quant_dense_1/ste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2.
,sequential/quant_dense_1/ste_sign_8/Identity 
-sequential/quant_dense_1/ste_sign_8/IdentityN	IdentityN.sequential/quant_dense_1/ste_sign_8/Sign_1:y:04sequential/batch_normalization_3/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-169470*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2/
-sequential/quant_dense_1/ste_sign_8/IdentityNЏ
.sequential/quant_dense_1/MatMul/ReadVariableOpReadVariableOp7sequential_quant_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype020
.sequential/quant_dense_1/MatMul/ReadVariableOpЌ
/sequential/quant_dense_1/MatMul/ste_sign_7/SignSign6sequential/quant_dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА21
/sequential/quant_dense_1/MatMul/ste_sign_7/Sign©
0sequential/quant_dense_1/MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=22
0sequential/quant_dense_1/MatMul/ste_sign_7/add/yД
.sequential/quant_dense_1/MatMul/ste_sign_7/addAddV23sequential/quant_dense_1/MatMul/ste_sign_7/Sign:y:09sequential/quant_dense_1/MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
АА20
.sequential/quant_dense_1/MatMul/ste_sign_7/addЌ
1sequential/quant_dense_1/MatMul/ste_sign_7/Sign_1Sign2sequential/quant_dense_1/MatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
АА23
1sequential/quant_dense_1/MatMul/ste_sign_7/Sign_1Ў
3sequential/quant_dense_1/MatMul/ste_sign_7/IdentityIdentity5sequential/quant_dense_1/MatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
АА25
3sequential/quant_dense_1/MatMul/ste_sign_7/Identity—
4sequential/quant_dense_1/MatMul/ste_sign_7/IdentityN	IdentityN5sequential/quant_dense_1/MatMul/ste_sign_7/Sign_1:y:06sequential/quant_dense_1/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-169480*,
_output_shapes
:
АА:
АА26
4sequential/quant_dense_1/MatMul/ste_sign_7/IdentityNц
sequential/quant_dense_1/MatMulMatMul6sequential/quant_dense_1/ste_sign_8/IdentityN:output:0=sequential/quant_dense_1/MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
sequential/quant_dense_1/MatMul†
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
.sequential/batch_normalization_4/batchnorm/mulэ
0sequential/batch_normalization_4/batchnorm/mul_1Mul)sequential/quant_dense_1/MatMul:product:02sequential/batch_normalization_4/batchnorm/mul:z:0*
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
)sequential/quant_dense_2/ste_sign_10/SignSign4sequential/batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2+
)sequential/quant_dense_2/ste_sign_10/SignЭ
*sequential/quant_dense_2/ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2,
*sequential/quant_dense_2/ste_sign_10/add/yф
(sequential/quant_dense_2/ste_sign_10/addAddV2-sequential/quant_dense_2/ste_sign_10/Sign:y:03sequential/quant_dense_2/ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(sequential/quant_dense_2/ste_sign_10/add√
+sequential/quant_dense_2/ste_sign_10/Sign_1Sign,sequential/quant_dense_2/ste_sign_10/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2-
+sequential/quant_dense_2/ste_sign_10/Sign_1ќ
-sequential/quant_dense_2/ste_sign_10/IdentityIdentity/sequential/quant_dense_2/ste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2/
-sequential/quant_dense_2/ste_sign_10/IdentityЌ
.sequential/quant_dense_2/ste_sign_10/IdentityN	IdentityN/sequential/quant_dense_2/ste_sign_10/Sign_1:y:04sequential/batch_normalization_4/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-169508*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А20
.sequential/quant_dense_2/ste_sign_10/IdentityNў
.sequential/quant_dense_2/MatMul/ReadVariableOpReadVariableOp7sequential_quant_dense_2_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype020
.sequential/quant_dense_2/MatMul/ReadVariableOpћ
/sequential/quant_dense_2/MatMul/ste_sign_9/SignSign6sequential/quant_dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А21
/sequential/quant_dense_2/MatMul/ste_sign_9/Sign©
0sequential/quant_dense_2/MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=22
0sequential/quant_dense_2/MatMul/ste_sign_9/add/yГ
.sequential/quant_dense_2/MatMul/ste_sign_9/addAddV23sequential/quant_dense_2/MatMul/ste_sign_9/Sign:y:09sequential/quant_dense_2/MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	А20
.sequential/quant_dense_2/MatMul/ste_sign_9/addћ
1sequential/quant_dense_2/MatMul/ste_sign_9/Sign_1Sign2sequential/quant_dense_2/MatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	А23
1sequential/quant_dense_2/MatMul/ste_sign_9/Sign_1„
3sequential/quant_dense_2/MatMul/ste_sign_9/IdentityIdentity5sequential/quant_dense_2/MatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	А25
3sequential/quant_dense_2/MatMul/ste_sign_9/Identityѕ
4sequential/quant_dense_2/MatMul/ste_sign_9/IdentityN	IdentityN5sequential/quant_dense_2/MatMul/ste_sign_9/Sign_1:y:06sequential/quant_dense_2/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-169518**
_output_shapes
:	А:	А26
4sequential/quant_dense_2/MatMul/ste_sign_9/IdentityNц
sequential/quant_dense_2/MatMulMatMul7sequential/quant_dense_2/ste_sign_10/IdentityN:output:0=sequential/quant_dense_2/MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€2!
sequential/quant_dense_2/MatMul†
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
:*
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
:20
.sequential/batch_normalization_5/batchnorm/add∆
0sequential/batch_normalization_5/batchnorm/RsqrtRsqrt2sequential/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:22
0sequential/batch_normalization_5/batchnorm/RsqrtБ
=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpFsequential_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential/batch_normalization_5/batchnorm/mul/ReadVariableOpЙ
.sequential/batch_normalization_5/batchnorm/mulMul4sequential/batch_normalization_5/batchnorm/Rsqrt:y:0Esequential/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.sequential/batch_normalization_5/batchnorm/mulь
0sequential/batch_normalization_5/batchnorm/mul_1Mul)sequential/quant_dense_2/MatMul:product:02sequential/batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
0sequential/batch_normalization_5/batchnorm/mul_1ы
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_1Й
0sequential/batch_normalization_5/batchnorm/mul_2MulCsequential/batch_normalization_5/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:22
0sequential/batch_normalization_5/batchnorm/mul_2ы
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpDsequential_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02=
;sequential/batch_normalization_5/batchnorm/ReadVariableOp_2З
.sequential/batch_normalization_5/batchnorm/subSubCsequential/batch_normalization_5/batchnorm/ReadVariableOp_2:value:04sequential/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:20
.sequential/batch_normalization_5/batchnorm/subЙ
0sequential/batch_normalization_5/batchnorm/add_1AddV24sequential/batch_normalization_5/batchnorm/mul_1:z:02sequential/batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
0sequential/batch_normalization_5/batchnorm/add_1±
sequential/activation/SoftmaxSoftmax4sequential/batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential/activation/Softmax 
IdentityIdentity'sequential/activation/Softmax:softmax:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1A^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_2/ReadVariableOp2^sequential/batch_normalization_2/ReadVariableOp_1:^sequential/batch_normalization_3/batchnorm/ReadVariableOp<^sequential/batch_normalization_3/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_3/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_3/batchnorm/mul/ReadVariableOp:^sequential/batch_normalization_4/batchnorm/ReadVariableOp<^sequential/batch_normalization_4/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_4/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_4/batchnorm/mul/ReadVariableOp:^sequential/batch_normalization_5/batchnorm/ReadVariableOp<^sequential/batch_normalization_5/batchnorm/ReadVariableOp_1<^sequential/batch_normalization_5/batchnorm/ReadVariableOp_2>^sequential/batch_normalization_5/batchnorm/mul/ReadVariableOp.^sequential/quant_conv2d/Conv2D/ReadVariableOp0^sequential/quant_conv2d_1/Conv2D/ReadVariableOp0^sequential/quant_conv2d_2/Conv2D/ReadVariableOp-^sequential/quant_dense/MatMul/ReadVariableOp/^sequential/quant_dense_1/MatMul/ReadVariableOp/^sequential/quant_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

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
И
И
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_170450

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
≈	
d
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_169753

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identityв
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-169744*n
_output_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
	IdentityNД

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_170151

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
Ж
s
-__inference_quant_conv2d_layer_call_fn_169585

inputs
unknown
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1695772
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

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
м
©
6__inference_batch_normalization_3_layer_call_fn_173094

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1702982
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
≠%
Ь
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_170748

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_170733
assignmovingavg_1_170740
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
8:8:8:8:8:*
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
loc:@AssignMovingAvg/170733*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/170733*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_170733*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/170733*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/170733*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_170733AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/170733*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/170740*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170740*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_170740*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170740*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170740*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_170740AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/170740*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€A
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€A
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
ф
ф
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172727

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
8:8:8:8:8:*
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
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€A
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
б
D
(__inference_flatten_layer_call_fn_172940

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1709082
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€8:W S
/
_output_shapes
:€€€€€€€€€8
 
_user_specified_nameinputs
И
©
6__inference_batch_normalization_1_layer_call_fn_172740

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
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1707482
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€A
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€A
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
ћ
І
4__inference_batch_normalization_layer_call_fn_172482

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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1696792
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_173068

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
м
L
0__inference_max_pooling2d_2_layer_call_fn_170157

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
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1701512
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
6__inference_batch_normalization_3_layer_call_fn_173081

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1702622
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
Љ
ф
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172645

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_169924

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
цY
Ѕ
F__inference_sequential_layer_call_and_return_conditional_losses_171221
quant_conv2d_input
quant_conv2d_171143
batch_normalization_171146
batch_normalization_171148
batch_normalization_171150
batch_normalization_171152
quant_conv2d_1_171156 
batch_normalization_1_171159 
batch_normalization_1_171161 
batch_normalization_1_171163 
batch_normalization_1_171165
quant_conv2d_2_171169 
batch_normalization_2_171172 
batch_normalization_2_171174 
batch_normalization_2_171176 
batch_normalization_2_171178
quant_dense_171183 
batch_normalization_3_171186 
batch_normalization_3_171188 
batch_normalization_3_171190 
batch_normalization_3_171192
quant_dense_1_171195 
batch_normalization_4_171198 
batch_normalization_4_171200 
batch_normalization_4_171202 
batch_normalization_4_171204
quant_dense_2_171207 
batch_normalization_5_171210 
batch_normalization_5_171212 
batch_normalization_5_171214 
batch_normalization_5_171216
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ$quant_conv2d/StatefulPartitionedCallҐ&quant_conv2d_1/StatefulPartitionedCallҐ&quant_conv2d_2/StatefulPartitionedCallҐ#quant_dense/StatefulPartitionedCallҐ%quant_dense_1/StatefulPartitionedCallҐ%quant_dense_2/StatefulPartitionedCallв
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputquant_conv2d_171143*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В8*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1695772&
$quant_conv2d/StatefulPartitionedCallр
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_171146batch_normalization_171148batch_normalization_171150batch_normalization_171152*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1706752-
+batch_normalization/StatefulPartitionedCallЎ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1697312
max_pooling2d/PartitionedCallэ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_171156*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1697872(
&quant_conv2d_1/StatefulPartitionedCall€
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_171159batch_normalization_1_171161batch_normalization_1_171163batch_normalization_1_171165*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1707702/
-batch_normalization_1/StatefulPartitionedCallа
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1699412!
max_pooling2d_1/PartitionedCall€
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_171169*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1699972(
&quant_conv2d_2/StatefulPartitionedCall€
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_171172batch_normalization_2_171174batch_normalization_2_171176batch_normalization_2_171178*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1708652/
-batch_normalization_2/StatefulPartitionedCallа
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1701512!
max_pooling2d_2/PartitionedCall≥
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1709082
flatten/PartitionedCallд
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_171183*
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
G__inference_quant_dense_layer_call_and_return_conditional_losses_1709392%
#quant_dense/StatefulPartitionedCallх
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_171186batch_normalization_3_171188batch_normalization_3_171190batch_normalization_3_171192*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1702982/
-batch_normalization_3/StatefulPartitionedCallВ
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_171195*
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
GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1710092'
%quant_dense_1/StatefulPartitionedCallч
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_171198batch_normalization_4_171200batch_normalization_4_171202batch_normalization_4_171204*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1704502/
-batch_normalization_4/StatefulPartitionedCallБ
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_171207*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1710792'
%quant_dense_2/StatefulPartitionedCallц
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_171210batch_normalization_5_171212batch_normalization_5_171214batch_normalization_5_171216*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1706022/
-batch_normalization_5/StatefulPartitionedCall…
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1711312
activation/PartitionedCallД
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

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
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall:d `
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
и
J
.__inference_max_pooling2d_layer_call_fn_169737

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
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1697312
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
Ґ
t
.__inference_quant_dense_2_layer_call_fn_173278

inputs
unknown
identityИҐStatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1710792
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

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
Ї
т
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172469

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
6__inference_batch_normalization_4_layer_call_fn_173235

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1704142
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
х%
Ь
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_169889

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_169874
assignmovingavg_1_169881
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
loc:@AssignMovingAvg/169874*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/169874*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_169874*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/169874*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/169874*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_169874AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/169874*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/169881*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/169881*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_169881*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/169881*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/169881*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_169881AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/169881*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
+__inference_ste_sign_2_layer_call_fn_173463

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_1697532
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
_user_specified_nameinputs
И
И
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_173222

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
Б
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_169941

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
И
©
6__inference_batch_normalization_1_layer_call_fn_172753

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
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1707702
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€A
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€A
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
х%
Ь
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172623

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_172608
assignmovingavg_1_172615
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
loc:@AssignMovingAvg/172608*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/172608*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_172608*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/172608*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/172608*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_172608AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/172608*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/172615*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172615*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_172615*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172615*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172615*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_172615AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/172615*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
+__inference_sequential_layer_call_fn_172336

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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1713052
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172799

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_172784
assignmovingavg_1_172791
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
loc:@AssignMovingAvg/172784*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/172784*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_172784*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/172784*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/172784*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_172784AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/172784*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/172791*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172791*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_172791*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172791*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172791*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_172791AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/172791*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
6__inference_batch_normalization_5_layer_call_fn_173402

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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1706022
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_170099

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_170084
assignmovingavg_1_170091
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
loc:@AssignMovingAvg/170084*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/170084*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_170084*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/170084*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/170084*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_170084AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/170084*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/170091*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170091*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_170091*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170091*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170091*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_170091AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/170091*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЄ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
)__inference_ste_sign_layer_call_fn_173429

inputs
identityА
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
D__inference_ste_sign_layer_call_and_return_conditional_losses_1695662
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
—
G
+__inference_ste_sign_3_layer_call_fn_173480

inputs
identityВ
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
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_1699862
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
И
І
4__inference_batch_normalization_layer_call_fn_172577

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1706752
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€В82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€В8::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€В8
 
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_170865

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 8:8:8:8:8:*
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
:€€€€€€€€€ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€ 8
 
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
ѓ%
Ъ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_170653

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_170638
assignmovingavg_1_170645
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
Const_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*L
_output_shapes:
8:€€€€€€€€€В8:8:8:8:8:*
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
loc:@AssignMovingAvg/170638*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/170638*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_170638*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/170638*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/170638*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_170638AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/170638*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/170645*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170645*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_170645*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170645*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170645*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_170645AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/170645*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpІ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:€€€€€€€€€В82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€В8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€В8
 
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_170843

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_170828
assignmovingavg_1_170835
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 8:8:8:8:8:*
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
loc:@AssignMovingAvg/170828*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/170828*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_170828*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/170828*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/170828*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_170828AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/170828*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/170835*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170835*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_170835*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170835*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170835*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_170835AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/170835*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€ 8
 
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
ј»
Ё,
"__inference__traced_restore_173983
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
Identity_15Я
AssignVariableOp_15AssignVariableOp&assignvariableop_15_quant_dense_kernelIdentity_15:output:0*
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
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_quant_dense_1_kernelIdentity_20:output:0*
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
AssignVariableOp_25AssignVariableOp(assignvariableop_25_quant_dense_2_kernelIdentity_25:output:0*
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
Identity_48¶
AssignVariableOp_48AssignVariableOp-assignvariableop_48_adam_quant_dense_kernel_mIdentity_48:output:0*
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
Identity_51®
AssignVariableOp_51AssignVariableOp/assignvariableop_51_adam_quant_dense_1_kernel_mIdentity_51:output:0*
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
AssignVariableOp_54AssignVariableOp/assignvariableop_54_adam_quant_dense_2_kernel_mIdentity_54:output:0*
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
Identity_66¶
AssignVariableOp_66AssignVariableOp-assignvariableop_66_adam_quant_dense_kernel_vIdentity_66:output:0*
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
Identity_69®
AssignVariableOp_69AssignVariableOp/assignvariableop_69_adam_quant_dense_1_kernel_vIdentity_69:output:0*
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
AssignVariableOp_72AssignVariableOp/assignvariableop_72_adam_quant_dense_2_kernel_vIdentity_72:output:0*
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
цY
Ѕ
F__inference_sequential_layer_call_and_return_conditional_losses_171140
quant_conv2d_input
quant_conv2d_170617
batch_normalization_170702
batch_normalization_170704
batch_normalization_170706
batch_normalization_170708
quant_conv2d_1_170712 
batch_normalization_1_170797 
batch_normalization_1_170799 
batch_normalization_1_170801 
batch_normalization_1_170803
quant_conv2d_2_170807 
batch_normalization_2_170892 
batch_normalization_2_170894 
batch_normalization_2_170896 
batch_normalization_2_170898
quant_dense_170948 
batch_normalization_3_170977 
batch_normalization_3_170979 
batch_normalization_3_170981 
batch_normalization_3_170983
quant_dense_1_171018 
batch_normalization_4_171047 
batch_normalization_4_171049 
batch_normalization_4_171051 
batch_normalization_4_171053
quant_dense_2_171088 
batch_normalization_5_171117 
batch_normalization_5_171119 
batch_normalization_5_171121 
batch_normalization_5_171123
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ$quant_conv2d/StatefulPartitionedCallҐ&quant_conv2d_1/StatefulPartitionedCallҐ&quant_conv2d_2/StatefulPartitionedCallҐ#quant_dense/StatefulPartitionedCallҐ%quant_dense_1/StatefulPartitionedCallҐ%quant_dense_2/StatefulPartitionedCallв
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallquant_conv2d_inputquant_conv2d_170617*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В8*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1695772&
$quant_conv2d/StatefulPartitionedCallр
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_170702batch_normalization_170704batch_normalization_170706batch_normalization_170708*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1706532-
+batch_normalization/StatefulPartitionedCallЎ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1697312
max_pooling2d/PartitionedCallэ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_170712*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1697872(
&quant_conv2d_1/StatefulPartitionedCall€
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_170797batch_normalization_1_170799batch_normalization_1_170801batch_normalization_1_170803*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1707482/
-batch_normalization_1/StatefulPartitionedCallа
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1699412!
max_pooling2d_1/PartitionedCall€
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_170807*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1699972(
&quant_conv2d_2/StatefulPartitionedCall€
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_170892batch_normalization_2_170894batch_normalization_2_170896batch_normalization_2_170898*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1708432/
-batch_normalization_2/StatefulPartitionedCallа
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1701512!
max_pooling2d_2/PartitionedCall≥
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1709082
flatten/PartitionedCallд
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_170948*
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
G__inference_quant_dense_layer_call_and_return_conditional_losses_1709392%
#quant_dense/StatefulPartitionedCallх
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_170977batch_normalization_3_170979batch_normalization_3_170981batch_normalization_3_170983*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1702622/
-batch_normalization_3/StatefulPartitionedCallВ
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_171018*
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
GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1710092'
%quant_dense_1/StatefulPartitionedCallч
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_171047batch_normalization_4_171049batch_normalization_4_171051batch_normalization_4_171053*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1704142/
-batch_normalization_4/StatefulPartitionedCallБ
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_171088*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1710792'
%quant_dense_2/StatefulPartitionedCallц
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_171117batch_normalization_5_171119batch_normalization_5_171121batch_normalization_5_171123*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1705662/
-batch_normalization_5/StatefulPartitionedCall…
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1711312
activation/PartitionedCallД
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

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
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall:d `
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
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_173441

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
 *Ќћћ=2
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

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-173432*8
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
Ч
§
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_171079

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
_gradient_op_typeCustomGradient-171059*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2
ste_sign_10/IdentityNО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpБ
MatMul/ste_sign_9/SignSignMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
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
:	А2
MatMul/ste_sign_9/addБ
MatMul/ste_sign_9/Sign_1SignMatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/Sign_1М
MatMul/ste_sign_9/IdentityIdentityMatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/Identityл
MatMul/ste_sign_9/IdentityN	IdentityNMatMul/ste_sign_9/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-171069**
_output_shapes
:	А:	А2
MatMul/ste_sign_9/IdentityNТ
MatMulMatMulste_sign_10/IdentityN:output:0$MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

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
И
И
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_170298

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
–
©
6__inference_batch_normalization_2_layer_call_fn_172847

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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1701342
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
+__inference_ste_sign_4_layer_call_fn_173497

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_1699632
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
_user_specified_nameinputs
О
§
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_173117

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
_gradient_op_typeCustomGradient-173097*<
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
_gradient_op_typeCustomGradient-173107*,
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
D__inference_ste_sign_layer_call_and_return_conditional_losses_173424

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
 *Ќћћ=2
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

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-173415*8
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
–
©
6__inference_batch_normalization_1_layer_call_fn_172671

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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1699242
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
О
§
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_171009

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
_gradient_op_typeCustomGradient-170989*<
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
_gradient_op_typeCustomGradient-170999*,
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
§
t
.__inference_quant_dense_1_layer_call_fn_173124

inputs
unknown
identityИҐStatefulPartitionedCall©
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
GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1710092
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
’0
»
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_173045

inputs
assignmovingavg_173020
assignmovingavg_1_173026)
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
loc:@AssignMovingAvg/173020*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_173020*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpƒ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/173020*
_output_shapes	
:А2
AssignMovingAvg/subї
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/173020*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_173020AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/173020*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/173026*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_173026*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpќ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173026*
_output_shapes	
:А2
AssignMovingAvg_1/sub≈
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/173026*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_173026AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/173026*
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
ц
т
O__inference_batch_normalization_layer_call_and_return_conditional_losses_170675

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ћ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:€€€€€€€€€В8:8:8:8:8:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constџ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:€€€€€€€€€В82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€В8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€В8
 
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
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_169986

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
 *Ќћћ=2
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

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-169977*8
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
—
d
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_169776

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
 *Ќћћ=2
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

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-169767*8
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
И
І
4__inference_batch_normalization_layer_call_fn_172564

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1706532
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€В82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€В8::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€В8
 
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172705

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_172690
assignmovingavg_1_172697
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
8:8:8:8:8:*
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
loc:@AssignMovingAvg/172690*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xѓ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/172690*
_output_shapes
: 2
AssignMovingAvg/subУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_172690*
_output_shapes
:8*
dtype02 
AssignMovingAvg/ReadVariableOpћ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/172690*
_output_shapes
:82
AssignMovingAvg/sub_1µ
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/172690*
_output_shapes
:82
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_172690AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/172690*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/172697*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/xЈ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172697*
_output_shapes
: 2
AssignMovingAvg_1/subЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_172697*
_output_shapes
:8*
dtype02"
 AssignMovingAvg_1/ReadVariableOpЎ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172697*
_output_shapes
:82
AssignMovingAvg_1/sub_1њ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/172697*
_output_shapes
:82
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_172697AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/172697*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp¶
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€A
82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€A
8::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€A
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
Ч
§
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_173271

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
_gradient_op_typeCustomGradient-173251*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2
ste_sign_10/IdentityNО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpБ
MatMul/ste_sign_9/SignSignMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
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
:	А2
MatMul/ste_sign_9/addБ
MatMul/ste_sign_9/Sign_1SignMatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/Sign_1М
MatMul/ste_sign_9/IdentityIdentityMatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	А2
MatMul/ste_sign_9/Identityл
MatMul/ste_sign_9/IdentityN	IdentityNMatMul/ste_sign_9/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-173261**
_output_shapes
:	А:	А2
MatMul/ste_sign_9/IdentityNТ
MatMulMatMulste_sign_10/IdentityN:output:0$MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

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
“Y
µ
F__inference_sequential_layer_call_and_return_conditional_losses_171305

inputs
quant_conv2d_171227
batch_normalization_171230
batch_normalization_171232
batch_normalization_171234
batch_normalization_171236
quant_conv2d_1_171240 
batch_normalization_1_171243 
batch_normalization_1_171245 
batch_normalization_1_171247 
batch_normalization_1_171249
quant_conv2d_2_171253 
batch_normalization_2_171256 
batch_normalization_2_171258 
batch_normalization_2_171260 
batch_normalization_2_171262
quant_dense_171267 
batch_normalization_3_171270 
batch_normalization_3_171272 
batch_normalization_3_171274 
batch_normalization_3_171276
quant_dense_1_171279 
batch_normalization_4_171282 
batch_normalization_4_171284 
batch_normalization_4_171286 
batch_normalization_4_171288
quant_dense_2_171291 
batch_normalization_5_171294 
batch_normalization_5_171296 
batch_normalization_5_171298 
batch_normalization_5_171300
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ$quant_conv2d/StatefulPartitionedCallҐ&quant_conv2d_1/StatefulPartitionedCallҐ&quant_conv2d_2/StatefulPartitionedCallҐ#quant_dense/StatefulPartitionedCallҐ%quant_dense_1/StatefulPartitionedCallҐ%quant_dense_2/StatefulPartitionedCall÷
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsquant_conv2d_171227*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В8*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_1695772&
$quant_conv2d/StatefulPartitionedCallр
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0batch_normalization_171230batch_normalization_171232batch_normalization_171234batch_normalization_171236*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€В8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1706532-
+batch_normalization/StatefulPartitionedCallЎ
max_pooling2d/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_1697312
max_pooling2d/PartitionedCallэ
&quant_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0quant_conv2d_1_171240*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_1697872(
&quant_conv2d_1/StatefulPartitionedCall€
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_171243batch_normalization_1_171245batch_normalization_1_171247batch_normalization_1_171249*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€A
8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1707482/
-batch_normalization_1/StatefulPartitionedCallа
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1699412!
max_pooling2d_1/PartitionedCall€
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0quant_conv2d_2_171253*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_1699972(
&quant_conv2d_2/StatefulPartitionedCall€
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_171256batch_normalization_2_171258batch_normalization_2_171260batch_normalization_2_171262*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1708432/
-batch_normalization_2/StatefulPartitionedCallа
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1701512!
max_pooling2d_2/PartitionedCall≥
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1709082
flatten/PartitionedCallд
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0quant_dense_171267*
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
G__inference_quant_dense_layer_call_and_return_conditional_losses_1709392%
#quant_dense/StatefulPartitionedCallх
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,quant_dense/StatefulPartitionedCall:output:0batch_normalization_3_171270batch_normalization_3_171272batch_normalization_3_171274batch_normalization_3_171276*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1702622/
-batch_normalization_3/StatefulPartitionedCallВ
%quant_dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0quant_dense_1_171279*
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
GPU2*0J 8*R
fMRK
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_1710092'
%quant_dense_1/StatefulPartitionedCallч
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_1/StatefulPartitionedCall:output:0batch_normalization_4_171282batch_normalization_4_171284batch_normalization_4_171286batch_normalization_4_171288*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1704142/
-batch_normalization_4/StatefulPartitionedCallБ
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0quant_dense_2_171291*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_1710792'
%quant_dense_2/StatefulPartitionedCallц
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall.quant_dense_2/StatefulPartitionedCall:output:0batch_normalization_5_171294batch_normalization_5_171296batch_normalization_5_171298batch_normalization_5_171300*
Tin	
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1705662/
-batch_normalization_5/StatefulPartitionedCall…
activation/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1711312
activation/PartitionedCallД
IdentityIdentity#activation/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall%^quant_conv2d/StatefulPartitionedCall'^quant_conv2d_1/StatefulPartitionedCall'^quant_conv2d_2/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dense_1/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

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
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dense_1/StatefulPartitionedCall%quant_dense_1/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall:X T
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
–
©
6__inference_batch_normalization_2_layer_call_fn_172834

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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1700992
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_169714

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
÷є
„
F__inference_sequential_layer_call_and_return_conditional_losses_172271

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
identityИҐ3batch_normalization/FusedBatchNormV3/ReadVariableOpҐ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ.batch_normalization_3/batchnorm/ReadVariableOpҐ0batch_normalization_3/batchnorm/ReadVariableOp_1Ґ0batch_normalization_3/batchnorm/ReadVariableOp_2Ґ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐ.batch_normalization_4/batchnorm/ReadVariableOpҐ0batch_normalization_4/batchnorm/ReadVariableOp_1Ґ0batch_normalization_4/batchnorm/ReadVariableOp_2Ґ2batch_normalization_4/batchnorm/mul/ReadVariableOpҐ.batch_normalization_5/batchnorm/ReadVariableOpҐ0batch_normalization_5/batchnorm/ReadVariableOp_1Ґ0batch_normalization_5/batchnorm/ReadVariableOp_2Ґ2batch_normalization_5/batchnorm/mul/ReadVariableOpҐ"quant_conv2d/Conv2D/ReadVariableOpҐ$quant_conv2d_1/Conv2D/ReadVariableOpҐ$quant_conv2d_2/Conv2D/ReadVariableOpҐ!quant_dense/MatMul/ReadVariableOpҐ#quant_dense_1/MatMul/ReadVariableOpҐ#quant_dense_2/MatMul/ReadVariableOpЉ
"quant_conv2d/Conv2D/ReadVariableOpReadVariableOp+quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02$
"quant_conv2d/Conv2D/ReadVariableOpЂ
!quant_conv2d/Conv2D/ste_sign/SignSign*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:82#
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
:82"
 quant_conv2d/Conv2D/ste_sign/add©
#quant_conv2d/Conv2D/ste_sign/Sign_1Sign$quant_conv2d/Conv2D/ste_sign/add:z:0*
T0*&
_output_shapes
:82%
#quant_conv2d/Conv2D/ste_sign/Sign_1і
%quant_conv2d/Conv2D/ste_sign/IdentityIdentity'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*
T0*&
_output_shapes
:82'
%quant_conv2d/Conv2D/ste_sign/IdentityІ
&quant_conv2d/Conv2D/ste_sign/IdentityN	IdentityN'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-172049*8
_output_shapes&
$:8:82(
&quant_conv2d/Conv2D/ste_sign/IdentityN–
quant_conv2d/Conv2DConv2Dinputs/quant_conv2d/Conv2D/ste_sign/IdentityN:output:0*
T0*0
_output_shapes
:€€€€€€€€€В8*
paddingSAME*
strides
2
quant_conv2d/Conv2DЖ
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
:8*
dtype02$
"batch_normalization/ReadVariableOpґ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:8*
dtype02&
$batch_normalization/ReadVariableOp_1г
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpй
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ў
$batch_normalization/FusedBatchNormV3FusedBatchNormV3quant_conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:€€€€€€€€€В8:8:8:8:8:*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3{
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization/Const–
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€A
8*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolҐ
quant_conv2d_1/ste_sign_2/SignSignmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
82 
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
82
quant_conv2d_1/ste_sign_2/add©
 quant_conv2d_1/ste_sign_2/Sign_1Sign!quant_conv2d_1/ste_sign_2/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€A
82"
 quant_conv2d_1/ste_sign_2/Sign_1і
"quant_conv2d_1/ste_sign_2/IdentityIdentity$quant_conv2d_1/ste_sign_2/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€A
82$
"quant_conv2d_1/ste_sign_2/Identity§
#quant_conv2d_1/ste_sign_2/IdentityN	IdentityN$quant_conv2d_1/ste_sign_2/Sign_1:y:0max_pooling2d/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-172077*J
_output_shapes8
6:€€€€€€€€€A
8:€€€€€€€€€A
82%
#quant_conv2d_1/ste_sign_2/IdentityN¬
$quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_1/Conv2D/ReadVariableOpµ
%quant_conv2d_1/Conv2D/ste_sign_1/SignSign,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:882'
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
:882&
$quant_conv2d_1/Conv2D/ste_sign_1/addµ
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1Sign(quant_conv2d_1/Conv2D/ste_sign_1/add:z:0*
T0*&
_output_shapes
:882)
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1ј
)quant_conv2d_1/Conv2D/ste_sign_1/IdentityIdentity+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0*
T0*&
_output_shapes
:882+
)quant_conv2d_1/Conv2D/ste_sign_1/Identityµ
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityN	IdentityN+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-172087*8
_output_shapes&
$:88:882,
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityNэ
quant_conv2d_1/Conv2DConv2D,quant_conv2d_1/ste_sign_2/IdentityN:output:03quant_conv2d_1/Conv2D/ste_sign_1/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
8*
paddingSAME*
strides
2
quant_conv2d_1/Conv2DК
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
:8*
dtype02&
$batch_normalization_1/ReadVariableOpЉ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:8*
dtype02(
&batch_normalization_1/ReadVariableOp_1й
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ж
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3quant_conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
8:8:8:8:8:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_1/Const÷
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€ 8*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool§
quant_conv2d_2/ste_sign_4/SignSign max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 82 
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
:€€€€€€€€€ 82
quant_conv2d_2/ste_sign_4/add©
 quant_conv2d_2/ste_sign_4/Sign_1Sign!quant_conv2d_2/ste_sign_4/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€ 82"
 quant_conv2d_2/ste_sign_4/Sign_1і
"quant_conv2d_2/ste_sign_4/IdentityIdentity$quant_conv2d_2/ste_sign_4/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 82$
"quant_conv2d_2/ste_sign_4/Identity¶
#quant_conv2d_2/ste_sign_4/IdentityN	IdentityN$quant_conv2d_2/ste_sign_4/Sign_1:y:0 max_pooling2d_1/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-172115*J
_output_shapes8
6:€€€€€€€€€ 8:€€€€€€€€€ 82%
#quant_conv2d_2/ste_sign_4/IdentityN¬
$quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_2/Conv2D/ReadVariableOpµ
%quant_conv2d_2/Conv2D/ste_sign_3/SignSign,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:882'
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
:882&
$quant_conv2d_2/Conv2D/ste_sign_3/addµ
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1Sign(quant_conv2d_2/Conv2D/ste_sign_3/add:z:0*
T0*&
_output_shapes
:882)
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1ј
)quant_conv2d_2/Conv2D/ste_sign_3/IdentityIdentity+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0*
T0*&
_output_shapes
:882+
)quant_conv2d_2/Conv2D/ste_sign_3/Identityµ
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityN	IdentityN+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-172125*8
_output_shapes&
$:88:882,
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityNэ
quant_conv2d_2/Conv2DConv2D,quant_conv2d_2/ste_sign_4/IdentityN:output:03quant_conv2d_2/Conv2D/ste_sign_3/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 8*
paddingSAME*
strides
2
quant_conv2d_2/Conv2DК
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
:8*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:8*
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ж
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3quant_conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 8:8:8:8:8:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
batch_normalization_2/Const÷
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€8*
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
valueB"€€€€   2
flatten/ConstЪ
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/ReshapeП
quant_dense/ste_sign_6/SignSignflatten/Reshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_6/SignБ
quant_dense/ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
quant_dense/ste_sign_6/add/yЉ
quant_dense/ste_sign_6/addAddV2quant_dense/ste_sign_6/Sign:y:0%quant_dense/ste_sign_6/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_6/addЩ
quant_dense/ste_sign_6/Sign_1Signquant_dense/ste_sign_6/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_6/Sign_1§
quant_dense/ste_sign_6/IdentityIdentity!quant_dense/ste_sign_6/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
quant_dense/ste_sign_6/IdentityЗ
 quant_dense/ste_sign_6/IdentityN	IdentityN!quant_dense/ste_sign_6/Sign_1:y:0flatten/Reshape:output:0*
T
2*,
_gradient_op_typeCustomGradient-172155*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2"
 quant_dense/ste_sign_6/IdentityN≥
!quant_dense/MatMul/ReadVariableOpReadVariableOp*quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!quant_dense/MatMul/ReadVariableOp¶
"quant_dense/MatMul/ste_sign_5/SignSign)quant_dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"quant_dense/MatMul/ste_sign_5/SignП
#quant_dense/MatMul/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2%
#quant_dense/MatMul/ste_sign_5/add/y–
!quant_dense/MatMul/ste_sign_5/addAddV2&quant_dense/MatMul/ste_sign_5/Sign:y:0,quant_dense/MatMul/ste_sign_5/add/y:output:0*
T0* 
_output_shapes
:
АА2#
!quant_dense/MatMul/ste_sign_5/add¶
$quant_dense/MatMul/ste_sign_5/Sign_1Sign%quant_dense/MatMul/ste_sign_5/add:z:0*
T0* 
_output_shapes
:
АА2&
$quant_dense/MatMul/ste_sign_5/Sign_1±
&quant_dense/MatMul/ste_sign_5/IdentityIdentity(quant_dense/MatMul/ste_sign_5/Sign_1:y:0*
T0* 
_output_shapes
:
АА2(
&quant_dense/MatMul/ste_sign_5/IdentityЭ
'quant_dense/MatMul/ste_sign_5/IdentityN	IdentityN(quant_dense/MatMul/ste_sign_5/Sign_1:y:0)quant_dense/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-172165*,
_output_shapes
:
АА:
АА2)
'quant_dense/MatMul/ste_sign_5/IdentityN¬
quant_dense/MatMulMatMul)quant_dense/ste_sign_6/IdentityN:output:00quant_dense/MatMul/ste_sign_5/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/MatMulК
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
 batch_normalization_3/LogicalAnd’
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yб
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_3/batchnorm/add¶
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_3/batchnorm/Rsqrtб
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpё
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_3/batchnorm/mulѕ
%batch_normalization_3/batchnorm/mul_1Mulquant_dense/MatMul:product:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_3/batchnorm/mul_1џ
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1ё
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_3/batchnorm/mul_2џ
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2№
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_3/batchnorm/subё
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_3/batchnorm/add_1§
quant_dense_1/ste_sign_8/SignSign)batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense_1/ste_sign_8/SignЕ
quant_dense_1/ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2 
quant_dense_1/ste_sign_8/add/yƒ
quant_dense_1/ste_sign_8/addAddV2!quant_dense_1/ste_sign_8/Sign:y:0'quant_dense_1/ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense_1/ste_sign_8/addЯ
quant_dense_1/ste_sign_8/Sign_1Sign quant_dense_1/ste_sign_8/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
quant_dense_1/ste_sign_8/Sign_1™
!quant_dense_1/ste_sign_8/IdentityIdentity#quant_dense_1/ste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!quant_dense_1/ste_sign_8/IdentityЮ
"quant_dense_1/ste_sign_8/IdentityN	IdentityN#quant_dense_1/ste_sign_8/Sign_1:y:0)batch_normalization_3/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-172193*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2$
"quant_dense_1/ste_sign_8/IdentityNє
#quant_dense_1/MatMul/ReadVariableOpReadVariableOp,quant_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02%
#quant_dense_1/MatMul/ReadVariableOpђ
$quant_dense_1/MatMul/ste_sign_7/SignSign+quant_dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2&
$quant_dense_1/MatMul/ste_sign_7/SignУ
%quant_dense_1/MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2'
%quant_dense_1/MatMul/ste_sign_7/add/yЎ
#quant_dense_1/MatMul/ste_sign_7/addAddV2(quant_dense_1/MatMul/ste_sign_7/Sign:y:0.quant_dense_1/MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
АА2%
#quant_dense_1/MatMul/ste_sign_7/addђ
&quant_dense_1/MatMul/ste_sign_7/Sign_1Sign'quant_dense_1/MatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
АА2(
&quant_dense_1/MatMul/ste_sign_7/Sign_1Ј
(quant_dense_1/MatMul/ste_sign_7/IdentityIdentity*quant_dense_1/MatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
АА2*
(quant_dense_1/MatMul/ste_sign_7/Identity•
)quant_dense_1/MatMul/ste_sign_7/IdentityN	IdentityN*quant_dense_1/MatMul/ste_sign_7/Sign_1:y:0+quant_dense_1/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-172203*,
_output_shapes
:
АА:
АА2+
)quant_dense_1/MatMul/ste_sign_7/IdentityN 
quant_dense_1/MatMulMatMul+quant_dense_1/ste_sign_8/IdentityN:output:02quant_dense_1/MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense_1/MatMulК
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
#batch_normalization_4/batchnorm/mul—
%batch_normalization_4/batchnorm/mul_1Mulquant_dense_1/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
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
quant_dense_2/ste_sign_10/SignSign)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
quant_dense_2/ste_sign_10/SignЗ
quant_dense_2/ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2!
quant_dense_2/ste_sign_10/add/y»
quant_dense_2/ste_sign_10/addAddV2"quant_dense_2/ste_sign_10/Sign:y:0(quant_dense_2/ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense_2/ste_sign_10/addҐ
 quant_dense_2/ste_sign_10/Sign_1Sign!quant_dense_2/ste_sign_10/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 quant_dense_2/ste_sign_10/Sign_1≠
"quant_dense_2/ste_sign_10/IdentityIdentity$quant_dense_2/ste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"quant_dense_2/ste_sign_10/Identity°
#quant_dense_2/ste_sign_10/IdentityN	IdentityN$quant_dense_2/ste_sign_10/Sign_1:y:0)batch_normalization_4/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-172231*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2%
#quant_dense_2/ste_sign_10/IdentityNЄ
#quant_dense_2/MatMul/ReadVariableOpReadVariableOp,quant_dense_2_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02%
#quant_dense_2/MatMul/ReadVariableOpЂ
$quant_dense_2/MatMul/ste_sign_9/SignSign+quant_dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2&
$quant_dense_2/MatMul/ste_sign_9/SignУ
%quant_dense_2/MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2'
%quant_dense_2/MatMul/ste_sign_9/add/y„
#quant_dense_2/MatMul/ste_sign_9/addAddV2(quant_dense_2/MatMul/ste_sign_9/Sign:y:0.quant_dense_2/MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	А2%
#quant_dense_2/MatMul/ste_sign_9/addЂ
&quant_dense_2/MatMul/ste_sign_9/Sign_1Sign'quant_dense_2/MatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	А2(
&quant_dense_2/MatMul/ste_sign_9/Sign_1ґ
(quant_dense_2/MatMul/ste_sign_9/IdentityIdentity*quant_dense_2/MatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	А2*
(quant_dense_2/MatMul/ste_sign_9/Identity£
)quant_dense_2/MatMul/ste_sign_9/IdentityN	IdentityN*quant_dense_2/MatMul/ste_sign_9/Sign_1:y:0+quant_dense_2/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-172241**
_output_shapes
:	А:	А2+
)quant_dense_2/MatMul/ste_sign_9/IdentityN 
quant_dense_2/MatMulMatMul,quant_dense_2/ste_sign_10/IdentityN:output:02quant_dense_2/MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
quant_dense_2/MatMulК
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
:*
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
:2%
#batch_normalization_5/batchnorm/add•
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtа
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mul–
%batch_normalization_5/batchnorm/mul_1Mulquant_dense_2/MatMul:product:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%batch_normalization_5/batchnorm/mul_1Џ
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1Ё
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2Џ
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2џ
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/subЁ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%batch_normalization_5/batchnorm/add_1Р
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
activation/Softmaxх
IdentityIdentityactivation/Softmax:softmax:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp#^quant_conv2d/Conv2D/ReadVariableOp%^quant_conv2d_1/Conv2D/ReadVariableOp%^quant_conv2d_2/Conv2D/ReadVariableOp"^quant_dense/MatMul/ReadVariableOp$^quant_dense_1/MatMul/ReadVariableOp$^quant_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

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
ћ
І
4__inference_batch_normalization_layer_call_fn_172495

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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1697142
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172821

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
6__inference_batch_normalization_5_layer_call_fn_173389

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
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1705662
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_173458

inputs

identity_1h
SignSigninputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
addm
Sign_1Signadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
Sign_1x
IdentityIdentity
Sign_1:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identityв
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-173449*n
_output_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
	IdentityNД

Identity_1IdentityIdentityN:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
_user_specified_nameinputs
ф
ф
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172903

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 8:8:8:8:8:*
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
:€€€€€€€€€ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€ 8
 
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
’
G
+__inference_activation_layer_call_fn_173412

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_1711312
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
—
d
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_173475

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
 *Ќћћ=2
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

Identityђ
	IdentityN	IdentityN
Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-173466*8
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
Њэ
У
F__inference_sequential_layer_call_and_return_conditional_losses_172044

inputs/
+quant_conv2d_conv2d_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource.
*batch_normalization_assignmovingavg_1717630
,batch_normalization_assignmovingavg_1_1717701
-quant_conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resource0
,batch_normalization_1_assignmovingavg_1718132
.batch_normalization_1_assignmovingavg_1_1718201
-quant_conv2d_2_conv2d_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resource0
,batch_normalization_2_assignmovingavg_1718632
.batch_normalization_2_assignmovingavg_1_171870.
*quant_dense_matmul_readvariableop_resource0
,batch_normalization_3_assignmovingavg_1719102
.batch_normalization_3_assignmovingavg_1_171916?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource0
,quant_dense_1_matmul_readvariableop_resource0
,batch_normalization_4_assignmovingavg_1719642
.batch_normalization_4_assignmovingavg_1_171970?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource0
,quant_dense_2_matmul_readvariableop_resource0
,batch_normalization_5_assignmovingavg_1720182
.batch_normalization_5_assignmovingavg_1_172024?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource
identityИҐ7batch_normalization/AssignMovingAvg/AssignSubVariableOpҐ2batch_normalization/AssignMovingAvg/ReadVariableOpҐ9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpҐ4batch_normalization/AssignMovingAvg_1/ReadVariableOpҐ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґ9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐ9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_4/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_4/batchnorm/ReadVariableOpҐ2batch_normalization_4/batchnorm/mul/ReadVariableOpҐ9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_5/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_5/batchnorm/ReadVariableOpҐ2batch_normalization_5/batchnorm/mul/ReadVariableOpҐ"quant_conv2d/Conv2D/ReadVariableOpҐ$quant_conv2d_1/Conv2D/ReadVariableOpҐ$quant_conv2d_2/Conv2D/ReadVariableOpҐ!quant_dense/MatMul/ReadVariableOpҐ#quant_dense_1/MatMul/ReadVariableOpҐ#quant_dense_2/MatMul/ReadVariableOpЉ
"quant_conv2d/Conv2D/ReadVariableOpReadVariableOp+quant_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:8*
dtype02$
"quant_conv2d/Conv2D/ReadVariableOpЂ
!quant_conv2d/Conv2D/ste_sign/SignSign*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:82#
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
:82"
 quant_conv2d/Conv2D/ste_sign/add©
#quant_conv2d/Conv2D/ste_sign/Sign_1Sign$quant_conv2d/Conv2D/ste_sign/add:z:0*
T0*&
_output_shapes
:82%
#quant_conv2d/Conv2D/ste_sign/Sign_1і
%quant_conv2d/Conv2D/ste_sign/IdentityIdentity'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*
T0*&
_output_shapes
:82'
%quant_conv2d/Conv2D/ste_sign/IdentityІ
&quant_conv2d/Conv2D/ste_sign/IdentityN	IdentityN'quant_conv2d/Conv2D/ste_sign/Sign_1:y:0*quant_conv2d/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-171738*8
_output_shapes&
$:8:82(
&quant_conv2d/Conv2D/ste_sign/IdentityN–
quant_conv2d/Conv2DConv2Dinputs/quant_conv2d/Conv2D/ste_sign/IdentityN:output:0*
T0*0
_output_shapes
:€€€€€€€€€В8*
paddingSAME*
strides
2
quant_conv2d/Conv2DЖ
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
:8*
dtype02$
"batch_normalization/ReadVariableOpґ
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
batch_normalization/Const_1Ф
$batch_normalization/FusedBatchNormV3FusedBatchNormV3quant_conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0"batch_normalization/Const:output:0$batch_normalization/Const_1:output:0*
T0*
U0*L
_output_shapes:
8:€€€€€€€€€В8:8:8:8:8:*
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
1/loc:@batch_normalization/AssignMovingAvg/171763*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)batch_normalization/AssignMovingAvg/sub/xУ
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/171763*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/subѕ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_assignmovingavg_171763*
_output_shapes
:8*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp∞
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/171763*
_output_shapes
:82+
)batch_normalization/AssignMovingAvg/sub_1Щ
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/171763*
_output_shapes
:82)
'batch_normalization/AssignMovingAvg/mulщ
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_assignmovingavg_171763+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/171763*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpа
+batch_normalization/AssignMovingAvg_1/sub/xConst*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/171770*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization/AssignMovingAvg_1/sub/xЫ
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/171770*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/sub’
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_assignmovingavg_1_171770*
_output_shapes
:8*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpЉ
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/171770*
_output_shapes
:82-
+batch_normalization/AssignMovingAvg_1/sub_1£
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/171770*
_output_shapes
:82+
)batch_normalization/AssignMovingAvg_1/mulЕ
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_assignmovingavg_1_171770-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/171770*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp–
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€A
8*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolҐ
quant_conv2d_1/ste_sign_2/SignSignmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
82 
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
82
quant_conv2d_1/ste_sign_2/add©
 quant_conv2d_1/ste_sign_2/Sign_1Sign!quant_conv2d_1/ste_sign_2/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€A
82"
 quant_conv2d_1/ste_sign_2/Sign_1і
"quant_conv2d_1/ste_sign_2/IdentityIdentity$quant_conv2d_1/ste_sign_2/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€A
82$
"quant_conv2d_1/ste_sign_2/Identity§
#quant_conv2d_1/ste_sign_2/IdentityN	IdentityN$quant_conv2d_1/ste_sign_2/Sign_1:y:0max_pooling2d/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-171778*J
_output_shapes8
6:€€€€€€€€€A
8:€€€€€€€€€A
82%
#quant_conv2d_1/ste_sign_2/IdentityN¬
$quant_conv2d_1/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_1/Conv2D/ReadVariableOpµ
%quant_conv2d_1/Conv2D/ste_sign_1/SignSign,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:882'
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
:882&
$quant_conv2d_1/Conv2D/ste_sign_1/addµ
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1Sign(quant_conv2d_1/Conv2D/ste_sign_1/add:z:0*
T0*&
_output_shapes
:882)
'quant_conv2d_1/Conv2D/ste_sign_1/Sign_1ј
)quant_conv2d_1/Conv2D/ste_sign_1/IdentityIdentity+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0*
T0*&
_output_shapes
:882+
)quant_conv2d_1/Conv2D/ste_sign_1/Identityµ
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityN	IdentityN+quant_conv2d_1/Conv2D/ste_sign_1/Sign_1:y:0,quant_conv2d_1/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-171788*8
_output_shapes&
$:88:882,
*quant_conv2d_1/Conv2D/ste_sign_1/IdentityNэ
quant_conv2d_1/Conv2DConv2D,quant_conv2d_1/ste_sign_2/IdentityN:output:03quant_conv2d_1/Conv2D/ste_sign_1/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€A
8*
paddingSAME*
strides
2
quant_conv2d_1/Conv2DК
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
:8*
dtype02&
$batch_normalization_1/ReadVariableOpЉ
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
batch_normalization_1/ConstБ
batch_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_1/Const_1°
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3quant_conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0$batch_normalization_1/Const:output:0&batch_normalization_1/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€A
8:8:8:8:8:*
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
31loc:@batch_normalization_1/AssignMovingAvg/171813*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_1/AssignMovingAvg/sub/xЭ
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/171813*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/sub’
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_1_assignmovingavg_171813*
_output_shapes
:8*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpЇ
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/171813*
_output_shapes
:82-
+batch_normalization_1/AssignMovingAvg/sub_1£
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/171813*
_output_shapes
:82+
)batch_normalization_1/AssignMovingAvg/mulЕ
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_1_assignmovingavg_171813-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/171813*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/171820*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_1/AssignMovingAvg_1/sub/x•
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/171820*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/subџ
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_1_assignmovingavg_1_171820*
_output_shapes
:8*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp∆
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/171820*
_output_shapes
:82/
-batch_normalization_1/AssignMovingAvg_1/sub_1≠
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/171820*
_output_shapes
:82-
+batch_normalization_1/AssignMovingAvg_1/mulС
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_1_assignmovingavg_1_171820/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/171820*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp÷
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€ 8*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool§
quant_conv2d_2/ste_sign_4/SignSign max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 82 
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
:€€€€€€€€€ 82
quant_conv2d_2/ste_sign_4/add©
 quant_conv2d_2/ste_sign_4/Sign_1Sign!quant_conv2d_2/ste_sign_4/add:z:0*
T0*/
_output_shapes
:€€€€€€€€€ 82"
 quant_conv2d_2/ste_sign_4/Sign_1і
"quant_conv2d_2/ste_sign_4/IdentityIdentity$quant_conv2d_2/ste_sign_4/Sign_1:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 82$
"quant_conv2d_2/ste_sign_4/Identity¶
#quant_conv2d_2/ste_sign_4/IdentityN	IdentityN$quant_conv2d_2/ste_sign_4/Sign_1:y:0 max_pooling2d_1/MaxPool:output:0*
T
2*,
_gradient_op_typeCustomGradient-171828*J
_output_shapes8
6:€€€€€€€€€ 8:€€€€€€€€€ 82%
#quant_conv2d_2/ste_sign_4/IdentityN¬
$quant_conv2d_2/Conv2D/ReadVariableOpReadVariableOp-quant_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02&
$quant_conv2d_2/Conv2D/ReadVariableOpµ
%quant_conv2d_2/Conv2D/ste_sign_3/SignSign,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:882'
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
:882&
$quant_conv2d_2/Conv2D/ste_sign_3/addµ
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1Sign(quant_conv2d_2/Conv2D/ste_sign_3/add:z:0*
T0*&
_output_shapes
:882)
'quant_conv2d_2/Conv2D/ste_sign_3/Sign_1ј
)quant_conv2d_2/Conv2D/ste_sign_3/IdentityIdentity+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0*
T0*&
_output_shapes
:882+
)quant_conv2d_2/Conv2D/ste_sign_3/Identityµ
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityN	IdentityN+quant_conv2d_2/Conv2D/ste_sign_3/Sign_1:y:0,quant_conv2d_2/Conv2D/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-171838*8
_output_shapes&
$:88:882,
*quant_conv2d_2/Conv2D/ste_sign_3/IdentityNэ
quant_conv2d_2/Conv2DConv2D,quant_conv2d_2/ste_sign_4/IdentityN:output:03quant_conv2d_2/Conv2D/ste_sign_3/IdentityN:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 8*
paddingSAME*
strides
2
quant_conv2d_2/Conv2DК
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
:8*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
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
batch_normalization_2/ConstБ
batch_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_2/Const_1°
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3quant_conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0$batch_normalization_2/Const:output:0&batch_normalization_2/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ 8:8:8:8:8:*
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
31loc:@batch_normalization_2/AssignMovingAvg/171863*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_2/AssignMovingAvg/sub/xЭ
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/171863*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/sub’
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_2_assignmovingavg_171863*
_output_shapes
:8*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpЇ
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/171863*
_output_shapes
:82-
+batch_normalization_2/AssignMovingAvg/sub_1£
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/171863*
_output_shapes
:82+
)batch_normalization_2/AssignMovingAvg/mulЕ
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_2_assignmovingavg_171863-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/171863*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/171870*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_2/AssignMovingAvg_1/sub/x•
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/171870*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/subџ
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_2_assignmovingavg_1_171870*
_output_shapes
:8*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp∆
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/171870*
_output_shapes
:82/
-batch_normalization_2/AssignMovingAvg_1/sub_1≠
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/171870*
_output_shapes
:82-
+batch_normalization_2/AssignMovingAvg_1/mulС
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_2_assignmovingavg_1_171870/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/171870*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp÷
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€8*
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
valueB"€€€€   2
flatten/ConstЪ
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/ReshapeП
quant_dense/ste_sign_6/SignSignflatten/Reshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_6/SignБ
quant_dense/ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
quant_dense/ste_sign_6/add/yЉ
quant_dense/ste_sign_6/addAddV2quant_dense/ste_sign_6/Sign:y:0%quant_dense/ste_sign_6/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_6/addЩ
quant_dense/ste_sign_6/Sign_1Signquant_dense/ste_sign_6/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/ste_sign_6/Sign_1§
quant_dense/ste_sign_6/IdentityIdentity!quant_dense/ste_sign_6/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
quant_dense/ste_sign_6/IdentityЗ
 quant_dense/ste_sign_6/IdentityN	IdentityN!quant_dense/ste_sign_6/Sign_1:y:0flatten/Reshape:output:0*
T
2*,
_gradient_op_typeCustomGradient-171880*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2"
 quant_dense/ste_sign_6/IdentityN≥
!quant_dense/MatMul/ReadVariableOpReadVariableOp*quant_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!quant_dense/MatMul/ReadVariableOp¶
"quant_dense/MatMul/ste_sign_5/SignSign)quant_dense/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2$
"quant_dense/MatMul/ste_sign_5/SignП
#quant_dense/MatMul/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2%
#quant_dense/MatMul/ste_sign_5/add/y–
!quant_dense/MatMul/ste_sign_5/addAddV2&quant_dense/MatMul/ste_sign_5/Sign:y:0,quant_dense/MatMul/ste_sign_5/add/y:output:0*
T0* 
_output_shapes
:
АА2#
!quant_dense/MatMul/ste_sign_5/add¶
$quant_dense/MatMul/ste_sign_5/Sign_1Sign%quant_dense/MatMul/ste_sign_5/add:z:0*
T0* 
_output_shapes
:
АА2&
$quant_dense/MatMul/ste_sign_5/Sign_1±
&quant_dense/MatMul/ste_sign_5/IdentityIdentity(quant_dense/MatMul/ste_sign_5/Sign_1:y:0*
T0* 
_output_shapes
:
АА2(
&quant_dense/MatMul/ste_sign_5/IdentityЭ
'quant_dense/MatMul/ste_sign_5/IdentityN	IdentityN(quant_dense/MatMul/ste_sign_5/Sign_1:y:0)quant_dense/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-171890*,
_output_shapes
:
АА:
АА2)
'quant_dense/MatMul/ste_sign_5/IdentityN¬
quant_dense/MatMulMatMul)quant_dense/ste_sign_6/IdentityN:output:00quant_dense/MatMul/ste_sign_5/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense/MatMulК
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
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesи
"batch_normalization_3/moments/meanMeanquant_dense/MatMul:product:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2$
"batch_normalization_3/moments/meanњ
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:	А2,
*batch_normalization_3/moments/StopGradientэ
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencequant_dense/MatMul:product:03batch_normalization_3/moments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€А21
/batch_normalization_3/moments/SquaredDifferenceЊ
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indicesЛ
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2(
&batch_normalization_3/moments/variance√
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_3/moments/SqueezeЋ
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1а
+batch_normalization_3/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/171910*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2-
+batch_normalization_3/AssignMovingAvg/decay÷
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_3_assignmovingavg_171910*
_output_shapes	
:А*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp≤
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/171910*
_output_shapes	
:А2+
)batch_normalization_3/AssignMovingAvg/sub©
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/171910*
_output_shapes	
:А2+
)batch_normalization_3/AssignMovingAvg/mulЕ
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_3_assignmovingavg_171910-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/171910*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_3/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/171916*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2/
-batch_normalization_3/AssignMovingAvg_1/decay№
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_3_assignmovingavg_1_171916*
_output_shapes	
:А*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpЉ
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/171916*
_output_shapes	
:А2-
+batch_normalization_3/AssignMovingAvg_1/sub≥
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/171916*
_output_shapes	
:А2-
+batch_normalization_3/AssignMovingAvg_1/mulС
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_3_assignmovingavg_1_171916/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/171916*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yџ
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_3/batchnorm/add¶
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_3/batchnorm/Rsqrtб
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpё
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_3/batchnorm/mulѕ
%batch_normalization_3/batchnorm/mul_1Mulquant_dense/MatMul:product:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_3/batchnorm/mul_1‘
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_3/batchnorm/mul_2’
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpЏ
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_3/batchnorm/subё
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_3/batchnorm/add_1§
quant_dense_1/ste_sign_8/SignSign)batch_normalization_3/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense_1/ste_sign_8/SignЕ
quant_dense_1/ste_sign_8/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2 
quant_dense_1/ste_sign_8/add/yƒ
quant_dense_1/ste_sign_8/addAddV2!quant_dense_1/ste_sign_8/Sign:y:0'quant_dense_1/ste_sign_8/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense_1/ste_sign_8/addЯ
quant_dense_1/ste_sign_8/Sign_1Sign quant_dense_1/ste_sign_8/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
quant_dense_1/ste_sign_8/Sign_1™
!quant_dense_1/ste_sign_8/IdentityIdentity#quant_dense_1/ste_sign_8/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!quant_dense_1/ste_sign_8/IdentityЮ
"quant_dense_1/ste_sign_8/IdentityN	IdentityN#quant_dense_1/ste_sign_8/Sign_1:y:0)batch_normalization_3/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-171934*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2$
"quant_dense_1/ste_sign_8/IdentityNє
#quant_dense_1/MatMul/ReadVariableOpReadVariableOp,quant_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02%
#quant_dense_1/MatMul/ReadVariableOpђ
$quant_dense_1/MatMul/ste_sign_7/SignSign+quant_dense_1/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2&
$quant_dense_1/MatMul/ste_sign_7/SignУ
%quant_dense_1/MatMul/ste_sign_7/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2'
%quant_dense_1/MatMul/ste_sign_7/add/yЎ
#quant_dense_1/MatMul/ste_sign_7/addAddV2(quant_dense_1/MatMul/ste_sign_7/Sign:y:0.quant_dense_1/MatMul/ste_sign_7/add/y:output:0*
T0* 
_output_shapes
:
АА2%
#quant_dense_1/MatMul/ste_sign_7/addђ
&quant_dense_1/MatMul/ste_sign_7/Sign_1Sign'quant_dense_1/MatMul/ste_sign_7/add:z:0*
T0* 
_output_shapes
:
АА2(
&quant_dense_1/MatMul/ste_sign_7/Sign_1Ј
(quant_dense_1/MatMul/ste_sign_7/IdentityIdentity*quant_dense_1/MatMul/ste_sign_7/Sign_1:y:0*
T0* 
_output_shapes
:
АА2*
(quant_dense_1/MatMul/ste_sign_7/Identity•
)quant_dense_1/MatMul/ste_sign_7/IdentityN	IdentityN*quant_dense_1/MatMul/ste_sign_7/Sign_1:y:0+quant_dense_1/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-171944*,
_output_shapes
:
АА:
АА2+
)quant_dense_1/MatMul/ste_sign_7/IdentityN 
quant_dense_1/MatMulMatMul+quant_dense_1/ste_sign_8/IdentityN:output:02quant_dense_1/MatMul/ste_sign_7/IdentityN:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense_1/MatMulК
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
4batch_normalization_4/moments/mean/reduction_indicesк
"batch_normalization_4/moments/meanMeanquant_dense_1/MatMul:product:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2$
"batch_normalization_4/moments/meanњ
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	А2,
*batch_normalization_4/moments/StopGradient€
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencequant_dense_1/MatMul:product:03batch_normalization_4/moments/StopGradient:output:0*
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
31loc:@batch_normalization_4/AssignMovingAvg/171964*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2-
+batch_normalization_4/AssignMovingAvg/decay÷
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_4_assignmovingavg_171964*
_output_shapes	
:А*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp≤
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/171964*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/sub©
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/171964*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/mulЕ
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_4_assignmovingavg_171964-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/171964*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_4/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/171970*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2/
-batch_normalization_4/AssignMovingAvg_1/decay№
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_4_assignmovingavg_1_171970*
_output_shapes	
:А*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpЉ
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/171970*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/sub≥
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/171970*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/mulС
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_4_assignmovingavg_1_171970/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/171970*
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
#batch_normalization_4/batchnorm/mul—
%batch_normalization_4/batchnorm/mul_1Mulquant_dense_1/MatMul:product:0'batch_normalization_4/batchnorm/mul:z:0*
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
quant_dense_2/ste_sign_10/SignSign)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
quant_dense_2/ste_sign_10/SignЗ
quant_dense_2/ste_sign_10/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2!
quant_dense_2/ste_sign_10/add/y»
quant_dense_2/ste_sign_10/addAddV2"quant_dense_2/ste_sign_10/Sign:y:0(quant_dense_2/ste_sign_10/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
quant_dense_2/ste_sign_10/addҐ
 quant_dense_2/ste_sign_10/Sign_1Sign!quant_dense_2/ste_sign_10/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 quant_dense_2/ste_sign_10/Sign_1≠
"quant_dense_2/ste_sign_10/IdentityIdentity$quant_dense_2/ste_sign_10/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"quant_dense_2/ste_sign_10/Identity°
#quant_dense_2/ste_sign_10/IdentityN	IdentityN$quant_dense_2/ste_sign_10/Sign_1:y:0)batch_normalization_4/batchnorm/add_1:z:0*
T
2*,
_gradient_op_typeCustomGradient-171988*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2%
#quant_dense_2/ste_sign_10/IdentityNЄ
#quant_dense_2/MatMul/ReadVariableOpReadVariableOp,quant_dense_2_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02%
#quant_dense_2/MatMul/ReadVariableOpЂ
$quant_dense_2/MatMul/ste_sign_9/SignSign+quant_dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2&
$quant_dense_2/MatMul/ste_sign_9/SignУ
%quant_dense_2/MatMul/ste_sign_9/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2'
%quant_dense_2/MatMul/ste_sign_9/add/y„
#quant_dense_2/MatMul/ste_sign_9/addAddV2(quant_dense_2/MatMul/ste_sign_9/Sign:y:0.quant_dense_2/MatMul/ste_sign_9/add/y:output:0*
T0*
_output_shapes
:	А2%
#quant_dense_2/MatMul/ste_sign_9/addЂ
&quant_dense_2/MatMul/ste_sign_9/Sign_1Sign'quant_dense_2/MatMul/ste_sign_9/add:z:0*
T0*
_output_shapes
:	А2(
&quant_dense_2/MatMul/ste_sign_9/Sign_1ґ
(quant_dense_2/MatMul/ste_sign_9/IdentityIdentity*quant_dense_2/MatMul/ste_sign_9/Sign_1:y:0*
T0*
_output_shapes
:	А2*
(quant_dense_2/MatMul/ste_sign_9/Identity£
)quant_dense_2/MatMul/ste_sign_9/IdentityN	IdentityN*quant_dense_2/MatMul/ste_sign_9/Sign_1:y:0+quant_dense_2/MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-171998**
_output_shapes
:	А:	А2+
)quant_dense_2/MatMul/ste_sign_9/IdentityN 
quant_dense_2/MatMulMatMul,quant_dense_2/ste_sign_10/IdentityN:output:02quant_dense_2/MatMul/ste_sign_9/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
quant_dense_2/MatMulК
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
"batch_normalization_5/moments/meanMeanquant_dense_2/MatMul:product:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_5/moments/meanЊ
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_5/moments/StopGradientю
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencequant_dense_2/MatMul:product:03batch_normalization_5/moments/StopGradient:output:0*
T0*'
_output_shapes
:€€€€€€€€€21
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

:*
	keep_dims(2(
&batch_normalization_5/moments/variance¬
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_5/moments/Squeeze 
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1а
+batch_normalization_5/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/172018*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2-
+batch_normalization_5/AssignMovingAvg/decay’
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_5_assignmovingavg_172018*
_output_shapes
:*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOp±
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/172018*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/sub®
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/172018*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/mulЕ
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_5_assignmovingavg_172018-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/172018*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpж
-batch_normalization_5/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/172024*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2/
-batch_normalization_5/AssignMovingAvg_1/decayџ
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_5_assignmovingavg_1_172024*
_output_shapes
:*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpї
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/172024*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/sub≤
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/172024*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/mulС
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_5_assignmovingavg_1_172024/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/172024*
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
:2%
#batch_normalization_5/batchnorm/add•
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtа
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mul–
%batch_normalization_5/batchnorm/mul_1Mulquant_dense_2/MatMul:product:0'batch_normalization_5/batchnorm/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%batch_normalization_5/batchnorm/mul_1”
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2‘
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpў
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/subЁ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%batch_normalization_5/batchnorm/add_1Р
activation/SoftmaxSoftmax)batch_normalization_5/batchnorm/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€2
activation/Softmaxе
IdentityIdentityactivation/Softmax:softmax:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp#^quant_conv2d/Conv2D/ReadVariableOp%^quant_conv2d_1/Conv2D/ReadVariableOp%^quant_conv2d_2/Conv2D/ReadVariableOp"^quant_dense/MatMul/ReadVariableOp$^quant_dense_1/MatMul/ReadVariableOp$^quant_dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

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
’0
»
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_170414

inputs
assignmovingavg_170389
assignmovingavg_1_170395)
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
loc:@AssignMovingAvg/170389*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_170389*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpƒ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/170389*
_output_shapes	
:А2
AssignMovingAvg/subї
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/170389*
_output_shapes	
:А2
AssignMovingAvg/mulБ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_170389AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/170389*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp§
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/170395*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_170395*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpќ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170395*
_output_shapes	
:А2
AssignMovingAvg_1/sub≈
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/170395*
_output_shapes	
:А2
AssignMovingAvg_1/mulН
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_170395AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/170395*
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
ї
_
C__inference_flatten_layer_call_and_return_conditional_losses_170908

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€8:W S
/
_output_shapes
:€€€€€€€€€8
 
_user_specified_nameinputs
О
•
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_169997

inputs"
conv2d_readvariableop_resource
identityИҐConv2D/ReadVariableOp≥
ste_sign_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_1699632
ste_sign_4/PartitionedCallІ
ste_sign_4/IdentityIdentity#ste_sign_4/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
ste_sign_4/IdentityХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02
Conv2D/ReadVariableOpљ
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
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_1699862#
!Conv2D/ste_sign_3/PartitionedCall°
Conv2D/ste_sign_3/IdentityIdentity*Conv2D/ste_sign_3/PartitionedCall:output:0*
T0*&
_output_shapes
:882
Conv2D/ste_sign_3/Identity—
Conv2DConv2Dste_sign_4/Identity:output:0#Conv2D/ste_sign_3/Identity:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*
paddingSAME*
strides
2
Conv2DХ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
_user_specified_nameinputs:

_output_shapes
: 
ц
т
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172551

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ћ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:€€€€€€€€€В8:8:8:8:8:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
Constџ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:€€€€€€€€€В82

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€В8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€В8
 
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_170134

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
:8*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:8*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:8*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:8*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:8:8:8:8:*
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
6__inference_batch_normalization_2_layer_call_fn_172916

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
:€€€€€€€€€ 8*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1708432
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 82

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€ 8::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 8
 
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
G__inference_quant_dense_layer_call_and_return_conditional_losses_170939

inputs"
matmul_readvariableop_resource
identityИҐMatMul/ReadVariableOpe
ste_sign_6/SignSigninputs*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_6/Signi
ste_sign_6/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
ste_sign_6/add/yМ
ste_sign_6/addAddV2ste_sign_6/Sign:y:0ste_sign_6/add/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_6/addu
ste_sign_6/Sign_1Signste_sign_6/add:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_6/Sign_1А
ste_sign_6/IdentityIdentityste_sign_6/Sign_1:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ste_sign_6/Identity—
ste_sign_6/IdentityN	IdentityNste_sign_6/Sign_1:y:0inputs*
T
2*,
_gradient_op_typeCustomGradient-170919*<
_output_shapes*
(:€€€€€€€€€А:€€€€€€€€€А2
ste_sign_6/IdentityNП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpВ
MatMul/ste_sign_5/SignSignMatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_5/Signw
MatMul/ste_sign_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
MatMul/ste_sign_5/add/y†
MatMul/ste_sign_5/addAddV2MatMul/ste_sign_5/Sign:y:0 MatMul/ste_sign_5/add/y:output:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_5/addВ
MatMul/ste_sign_5/Sign_1SignMatMul/ste_sign_5/add:z:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_5/Sign_1Н
MatMul/ste_sign_5/IdentityIdentityMatMul/ste_sign_5/Sign_1:y:0*
T0* 
_output_shapes
:
АА2
MatMul/ste_sign_5/Identityн
MatMul/ste_sign_5/IdentityN	IdentityNMatMul/ste_sign_5/Sign_1:y:0MatMul/ReadVariableOp:value:0*
T
2*,
_gradient_op_typeCustomGradient-170929*,
_output_shapes
:
АА:
АА2
MatMul/ste_sign_5/IdentityNТ
MatMulMatMulste_sign_6/IdentityN:output:0$MatMul/ste_sign_5/IdentityN:output:0*
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
:€€€€€€€€€А:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: 
О
•
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_169787

inputs"
conv2d_readvariableop_resource
identityИҐConv2D/ReadVariableOp≥
ste_sign_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_1697532
ste_sign_2/PartitionedCallІ
ste_sign_2/IdentityIdentity#ste_sign_2/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82
ste_sign_2/IdentityХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:88*
dtype02
Conv2D/ReadVariableOpљ
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
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_1697762#
!Conv2D/ste_sign_1/PartitionedCall°
Conv2D/ste_sign_1/IdentityIdentity*Conv2D/ste_sign_1/PartitionedCall:output:0*
T0*&
_output_shapes
:882
Conv2D/ste_sign_1/Identity—
Conv2DConv2Dste_sign_2/Identity:output:0#Conv2D/ste_sign_1/Identity:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8*
paddingSAME*
strides
2
Conv2DХ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€82

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
 
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
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:фт
УЮ
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
т_default_save_signature
+у&call_and_return_all_conditional_losses
ф__call__"¶Ш
_tf_keras_sequentialЖШ{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "QuantConv2D", "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "QuantDense", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}], "build_input_shape": [null, 130, 20, 1]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "QuantConv2D", "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "QuantConv2D", "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "QuantDense", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "QuantDense", "config": {"name": "quant_dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}], "build_input_shape": [null, 130, 20, 1]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": ["accuracy", "sparse_categorical_accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0013894954463467002, "decay": 0.0, "beta_1": 0.949999988079071, "beta_2": 0.9990000128746033, "epsilon": 1e-08, "amsgrad": false}}}}
Ђ

kernel_quantizer

kernel
	variables
regularization_losses
trainable_variables
	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"ш
_tf_keras_layerё{"class_name": "QuantConv2D", "name": "quant_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 130, 20, 1], "config": {"name": "quant_conv2d", "trainable": true, "batch_input_shape": [null, 130, 20, 1], "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": null, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
∞
axis
	gamma
 beta
!moving_mean
"moving_variance
#	variables
$regularization_losses
%trainable_variables
&	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"Џ
_tf_keras_layerј{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 56}}}}
ы
'	variables
(regularization_losses
)trainable_variables
*	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"к
_tf_keras_layer–{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Б
+kernel_quantizer
,input_quantizer

-kernel
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"є	
_tf_keras_layerЯ	{"class_name": "QuantConv2D", "name": "quant_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_conv2d_1", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 56}}}}
і
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7	variables
8regularization_losses
9trainable_variables
:	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"ё
_tf_keras_layerƒ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 56}}}}
€
;	variables
<regularization_losses
=trainable_variables
>	keras_api
+€&call_and_return_all_conditional_losses
А__call__"о
_tf_keras_layer‘{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Б
?kernel_quantizer
@input_quantizer

Akernel
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"є	
_tf_keras_layerЯ	{"class_name": "QuantConv2D", "name": "quant_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_conv2d_2", "trainable": true, "dtype": "float32", "filters": 56, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "pad_values": 0.0, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 56}}}}
і
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"ё
_tf_keras_layerƒ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 56}}}}
€
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"о
_tf_keras_layer‘{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ѓ
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
+З&call_and_return_all_conditional_losses
И__call__"Э
_tf_keras_layerГ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
т	
Wkernel_quantizer
Xinput_quantizer

Ykernel
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"™
_tf_keras_layerР{"class_name": "QuantDense", "name": "quant_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1792}}}}
µ
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
c	variables
dregularization_losses
etrainable_variables
f	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"я
_tf_keras_layer≈{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 384}}}}
х	
gkernel_quantizer
hinput_quantizer

ikernel
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"≠
_tf_keras_layerУ{"class_name": "QuantDense", "name": "quant_dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense_1", "trainable": true, "dtype": "float32", "units": 384, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}}
µ
naxis
	ogamma
pbeta
qmoving_mean
rmoving_variance
s	variables
tregularization_losses
utrainable_variables
v	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"я
_tf_keras_layer≈{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 384}}}}
ф	
wkernel_quantizer
xinput_quantizer

ykernel
z	variables
{regularization_losses
|trainable_variables
}	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"ђ
_tf_keras_layerТ{"class_name": "QuantDense", "name": "quant_dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "quant_dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "weight_clip", "config": {"clip_value": 1}}, "bias_constraint": null, "input_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}, "kernel_quantizer": {"class_name": "SteSign", "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}}
Ї
~axis
	gamma
	Аbeta
Бmoving_mean
Вmoving_variance
Г	variables
Дregularization_losses
Еtrainable_variables
Ж	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"Ё
_tf_keras_layer√{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.9, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 8}}}}
§
З	variables
Иregularization_losses
Йtrainable_variables
К	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"П
_tf_keras_layerх{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "softmax"}}
¬
	Лiter
Мbeta_1
Нbeta_2

Оdecay
Пlearning_ratemќmѕ m–-m—3m“4m”Am‘Gm’Hm÷Ym„_mЎ`mўimЏomџpm№ymЁmё	Аmяvаvб vв-vг3vд4vеAvжGvзHvиYvй_vк`vлivмovнpvоyvпvр	Аvс"
	optimizer
Й
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
А27
Б28
В29"
trackable_list_wrapper
 "
trackable_list_wrapper
І
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
А17"
trackable_list_wrapper
њ
	variables
regularization_losses
Рnon_trainable_variables
 Сlayer_regularization_losses
Тmetrics
Уlayers
trainable_variables
ф__call__
т_default_save_signature
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
-
Чserving_default"
signature_map
≠
Ф_custom_metrics
Х	variables
Цregularization_losses
Чtrainable_variables
Ш	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"В
_tf_keras_layerи{"class_name": "SteSign", "name": "ste_sign", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
-:+82quant_conv2d/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
	variables
regularization_losses
Щnon_trainable_variables
 Ъlayer_regularization_losses
Ыmetrics
Ьlayers
trainable_variables
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%82batch_normalization/gamma
&:$82batch_normalization/beta
/:-8 (2batch_normalization/moving_mean
3:18 (2#batch_normalization/moving_variance
<
0
 1
!2
"3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
°
#	variables
$regularization_losses
Эnon_trainable_variables
 Юlayer_regularization_losses
Яmetrics
†layers
%trainable_variables
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
'	variables
(regularization_losses
°non_trainable_variables
 Ґlayer_regularization_losses
£metrics
§layers
)trainable_variables
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
±
•_custom_metrics
¶	variables
Іregularization_losses
®trainable_variables
©	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_1", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
Ы
™	variables
Ђregularization_losses
ђtrainable_variables
≠	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_2", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
/:-882quant_conv2d_1/kernel
'
-0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
-0"
trackable_list_wrapper
°
.	variables
/regularization_losses
Ѓnon_trainable_variables
 ѓlayer_regularization_losses
∞metrics
±layers
0trainable_variables
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'82batch_normalization_1/gamma
(:&82batch_normalization_1/beta
1:/8 (2!batch_normalization_1/moving_mean
5:38 (2%batch_normalization_1/moving_variance
<
30
41
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
°
7	variables
8regularization_losses
≤non_trainable_variables
 ≥layer_regularization_losses
іmetrics
µlayers
9trainable_variables
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
;	variables
<regularization_losses
ґnon_trainable_variables
 Јlayer_regularization_losses
Єmetrics
єlayers
=trainable_variables
А__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
±
Ї_custom_metrics
ї	variables
Љregularization_losses
љtrainable_variables
Њ	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_3", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
Ы
њ	variables
јregularization_losses
Ѕtrainable_variables
¬	keras_api
+†&call_and_return_all_conditional_losses
°__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_4", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
/:-882quant_conv2d_2/kernel
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
A0"
trackable_list_wrapper
°
B	variables
Cregularization_losses
√non_trainable_variables
 ƒlayer_regularization_losses
≈metrics
∆layers
Dtrainable_variables
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'82batch_normalization_2/gamma
(:&82batch_normalization_2/beta
1:/8 (2!batch_normalization_2/moving_mean
5:38 (2%batch_normalization_2/moving_variance
<
G0
H1
I2
J3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
°
K	variables
Lregularization_losses
«non_trainable_variables
 »layer_regularization_losses
…metrics
 layers
Mtrainable_variables
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
O	variables
Pregularization_losses
Ћnon_trainable_variables
 ћlayer_regularization_losses
Ќmetrics
ќlayers
Qtrainable_variables
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
S	variables
Tregularization_losses
ѕnon_trainable_variables
 –layer_regularization_losses
—metrics
“layers
Utrainable_variables
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
±
”_custom_metrics
‘	variables
’regularization_losses
÷trainable_variables
„	keras_api
+Ґ&call_and_return_all_conditional_losses
£__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_5", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
Ы
Ў	variables
ўregularization_losses
Џtrainable_variables
џ	keras_api
+§&call_and_return_all_conditional_losses
•__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_6", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
&:$
АА2quant_dense/kernel
'
Y0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
Y0"
trackable_list_wrapper
°
Z	variables
[regularization_losses
№non_trainable_variables
 Ёlayer_regularization_losses
ёmetrics
яlayers
\trainable_variables
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_3/gamma
):'А2batch_normalization_3/beta
2:0А (2!batch_normalization_3/moving_mean
6:4А (2%batch_normalization_3/moving_variance
<
_0
`1
a2
b3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
°
c	variables
dregularization_losses
аnon_trainable_variables
 бlayer_regularization_losses
вmetrics
гlayers
etrainable_variables
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
±
д_custom_metrics
е	variables
жregularization_losses
зtrainable_variables
и	keras_api
+¶&call_and_return_all_conditional_losses
І__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_7", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
Ы
й	variables
кregularization_losses
лtrainable_variables
м	keras_api
+®&call_and_return_all_conditional_losses
©__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_8", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
(:&
АА2quant_dense_1/kernel
'
i0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
i0"
trackable_list_wrapper
°
j	variables
kregularization_losses
нnon_trainable_variables
 оlayer_regularization_losses
пmetrics
рlayers
ltrainable_variables
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_4/gamma
):'А2batch_normalization_4/beta
2:0А (2!batch_normalization_4/moving_mean
6:4А (2%batch_normalization_4/moving_variance
<
o0
p1
q2
r3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
°
s	variables
tregularization_losses
сnon_trainable_variables
 тlayer_regularization_losses
уmetrics
фlayers
utrainable_variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
±
х_custom_metrics
ц	variables
чregularization_losses
шtrainable_variables
щ	keras_api
+™&call_and_return_all_conditional_losses
Ђ__call__"Ж
_tf_keras_layerм{"class_name": "SteSign", "name": "ste_sign_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_9", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
Э
ъ	variables
ыregularization_losses
ьtrainable_variables
э	keras_api
+ђ&call_and_return_all_conditional_losses
≠__call__"И
_tf_keras_layerо{"class_name": "SteSign", "name": "ste_sign_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ste_sign_10", "trainable": true, "dtype": "float32", "clip_value": 1.0}}
':%	А2quant_dense_2/kernel
'
y0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
y0"
trackable_list_wrapper
°
z	variables
{regularization_losses
юnon_trainable_variables
 €layer_regularization_losses
Аmetrics
Бlayers
|trainable_variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
?
0
А1
Б2
В3"
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
А1"
trackable_list_wrapper
§
Г	variables
Дregularization_losses
Вnon_trainable_variables
 Гlayer_regularization_losses
Дmetrics
Еlayers
Еtrainable_variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
З	variables
Иregularization_losses
Жnon_trainable_variables
 Зlayer_regularization_losses
Иmetrics
Йlayers
Йtrainable_variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
Б10
В11"
trackable_list_wrapper
 "
trackable_list_wrapper
0
К0
Л1"
trackable_list_wrapper
Ю
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
§
Х	variables
Цregularization_losses
Мnon_trainable_variables
 Нlayer_regularization_losses
Оmetrics
Пlayers
Чtrainable_variables
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
¶	variables
Іregularization_losses
Рnon_trainable_variables
 Сlayer_regularization_losses
Тmetrics
Уlayers
®trainable_variables
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
™	variables
Ђregularization_losses
Фnon_trainable_variables
 Хlayer_regularization_losses
Цmetrics
Чlayers
ђtrainable_variables
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
ї	variables
Љregularization_losses
Шnon_trainable_variables
 Щlayer_regularization_losses
Ъmetrics
Ыlayers
љtrainable_variables
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
њ	variables
јregularization_losses
Ьnon_trainable_variables
 Эlayer_regularization_losses
Юmetrics
Яlayers
Ѕtrainable_variables
°__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
‘	variables
’regularization_losses
†non_trainable_variables
 °layer_regularization_losses
Ґmetrics
£layers
÷trainable_variables
£__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
Ў	variables
ўregularization_losses
§non_trainable_variables
 •layer_regularization_losses
¶metrics
Іlayers
Џtrainable_variables
•__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
е	variables
жregularization_losses
®non_trainable_variables
 ©layer_regularization_losses
™metrics
Ђlayers
зtrainable_variables
І__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
й	variables
кregularization_losses
ђnon_trainable_variables
 ≠layer_regularization_losses
Ѓmetrics
ѓlayers
лtrainable_variables
©__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
ц	variables
чregularization_losses
∞non_trainable_variables
 ±layer_regularization_losses
≤metrics
≥layers
шtrainable_variables
Ђ__call__
+™&call_and_return_all_conditional_losses
'™"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
ъ	variables
ыregularization_losses
іnon_trainable_variables
 µlayer_regularization_losses
ґmetrics
Јlayers
ьtrainable_variables
≠__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
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
0
Б0
В1"
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
£

Єtotal

єcount
Ї
_fn_kwargs
ї	variables
Љregularization_losses
љtrainable_variables
Њ	keras_api
+Ѓ&call_and_return_all_conditional_losses
ѓ__call__"е
_tf_keras_layerЋ{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
…

њtotal

јcount
Ѕ
_fn_kwargs
¬	variables
√regularization_losses
ƒtrainable_variables
≈	keras_api
+∞&call_and_return_all_conditional_losses
±__call__"Л
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
Є0
є1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
ї	variables
Љregularization_losses
∆non_trainable_variables
 «layer_regularization_losses
»metrics
…layers
љtrainable_variables
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
њ0
ј1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
§
¬	variables
√regularization_losses
 non_trainable_variables
 Ћlayer_regularization_losses
ћmetrics
Ќlayers
ƒtrainable_variables
±__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
0
Є0
є1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
њ0
ј1"
trackable_list_wrapper
 "
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
АА2Adam/quant_dense/kernel/m
/:-А2"Adam/batch_normalization_3/gamma/m
.:,А2!Adam/batch_normalization_3/beta/m
-:+
АА2Adam/quant_dense_1/kernel/m
/:-А2"Adam/batch_normalization_4/gamma/m
.:,А2!Adam/batch_normalization_4/beta/m
,:*	А2Adam/quant_dense_2/kernel/m
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
АА2Adam/quant_dense/kernel/v
/:-А2"Adam/batch_normalization_3/gamma/v
.:,А2!Adam/batch_normalization_3/beta/v
-:+
АА2Adam/quant_dense_1/kernel/v
/:-А2"Adam/batch_normalization_4/gamma/v
.:,А2!Adam/batch_normalization_4/beta/v
,:*	А2Adam/quant_dense_2/kernel/v
.:,2"Adam/batch_normalization_5/gamma/v
-:+2!Adam/batch_normalization_5/beta/v
у2р
!__inference__wrapped_model_169548 
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
ж2г
F__inference_sequential_layer_call_and_return_conditional_losses_172044
F__inference_sequential_layer_call_and_return_conditional_losses_172271
F__inference_sequential_layer_call_and_return_conditional_losses_171221
F__inference_sequential_layer_call_and_return_conditional_losses_171140ј
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
+__inference_sequential_layer_call_fn_171368
+__inference_sequential_layer_call_fn_171514
+__inference_sequential_layer_call_fn_172401
+__inference_sequential_layer_call_fn_172336ј
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
І2§
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_169577„
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
-__inference_quant_conv2d_layer_call_fn_169585„
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
ю2ы
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172469
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172551
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172447
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172529і
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
4__inference_batch_normalization_layer_call_fn_172482
4__inference_batch_normalization_layer_call_fn_172564
4__inference_batch_normalization_layer_call_fn_172495
4__inference_batch_normalization_layer_call_fn_172577і
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
±2Ѓ
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_169731а
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
.__inference_max_pooling2d_layer_call_fn_169737а
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
©2¶
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_169787„
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
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
О2Л
/__inference_quant_conv2d_1_layer_call_fn_169795„
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
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ж2Г
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172623
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172705
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172645
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172727і
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
6__inference_batch_normalization_1_layer_call_fn_172753
6__inference_batch_normalization_1_layer_call_fn_172671
6__inference_batch_normalization_1_layer_call_fn_172740
6__inference_batch_normalization_1_layer_call_fn_172658і
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
≥2∞
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_169941а
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
0__inference_max_pooling2d_1_layer_call_fn_169947а
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
©2¶
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_169997„
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
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
О2Л
/__inference_quant_conv2d_2_layer_call_fn_170005„
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
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ж2Г
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172881
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172821
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172903
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172799і
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
6__inference_batch_normalization_2_layer_call_fn_172834
6__inference_batch_normalization_2_layer_call_fn_172929
6__inference_batch_normalization_2_layer_call_fn_172847
6__inference_batch_normalization_2_layer_call_fn_172916і
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
≥2∞
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_170151а
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
0__inference_max_pooling2d_2_layer_call_fn_170157а
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
н2к
C__inference_flatten_layer_call_and_return_conditional_losses_172935Ґ
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
(__inference_flatten_layer_call_fn_172940Ґ
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
G__inference_quant_dense_layer_call_and_return_conditional_losses_172963Ґ
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
,__inference_quant_dense_layer_call_fn_172970Ґ
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_173068
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_173045і
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
6__inference_batch_normalization_3_layer_call_fn_173081
6__inference_batch_normalization_3_layer_call_fn_173094і
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
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_173117Ґ
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
.__inference_quant_dense_1_layer_call_fn_173124Ґ
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_173222
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_173199і
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
6__inference_batch_normalization_4_layer_call_fn_173248
6__inference_batch_normalization_4_layer_call_fn_173235і
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
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_173271Ґ
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
.__inference_quant_dense_2_layer_call_fn_173278Ґ
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_173376
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_173353і
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
6__inference_batch_normalization_5_layer_call_fn_173402
6__inference_batch_normalization_5_layer_call_fn_173389і
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
F__inference_activation_layer_call_and_return_conditional_losses_173407Ґ
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
+__inference_activation_layer_call_fn_173412Ґ
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
$__inference_signature_wrapper_171733quant_conv2d_input
о2л
D__inference_ste_sign_layer_call_and_return_conditional_losses_173424Ґ
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
)__inference_ste_sign_layer_call_fn_173429Ґ
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
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_173441Ґ
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
+__inference_ste_sign_1_layer_call_fn_173446Ґ
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
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_173458Ґ
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
+__inference_ste_sign_2_layer_call_fn_173463Ґ
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
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_173475Ґ
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
+__inference_ste_sign_3_layer_call_fn_173480Ґ
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
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_173492Ґ
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
+__inference_ste_sign_4_layer_call_fn_173497Ґ
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
 »
!__inference__wrapped_model_169548Ґ! !"-3456AGHIJYb_a`iroqpyВБАDҐA
:Ґ7
5К2
quant_conv2d_input€€€€€€€€€В
™ "7™4
2

activation$К!

activation€€€€€€€€€Ґ
F__inference_activation_layer_call_and_return_conditional_losses_173407X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ z
+__inference_activation_layer_call_fn_173412K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€м
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172623Ц3456MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ м
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172645Ц3456MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ «
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172705r3456;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€A
8
p
™ "-Ґ*
#К 
0€€€€€€€€€A
8
Ъ «
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_172727r3456;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€A
8
p 
™ "-Ґ*
#К 
0€€€€€€€€€A
8
Ъ ƒ
6__inference_batch_normalization_1_layer_call_fn_172658Й3456MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8ƒ
6__inference_batch_normalization_1_layer_call_fn_172671Й3456MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8Я
6__inference_batch_normalization_1_layer_call_fn_172740e3456;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€A
8
p
™ " К€€€€€€€€€A
8Я
6__inference_batch_normalization_1_layer_call_fn_172753e3456;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€A
8
p 
™ " К€€€€€€€€€A
8м
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172799ЦGHIJMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ м
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172821ЦGHIJMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ «
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172881rGHIJ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 8
p
™ "-Ґ*
#К 
0€€€€€€€€€ 8
Ъ «
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_172903rGHIJ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 8
p 
™ "-Ґ*
#К 
0€€€€€€€€€ 8
Ъ ƒ
6__inference_batch_normalization_2_layer_call_fn_172834ЙGHIJMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8ƒ
6__inference_batch_normalization_2_layer_call_fn_172847ЙGHIJMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8Я
6__inference_batch_normalization_2_layer_call_fn_172916eGHIJ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 8
p
™ " К€€€€€€€€€ 8Я
6__inference_batch_normalization_2_layer_call_fn_172929eGHIJ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 8
p 
™ " К€€€€€€€€€ 8є
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_173045dab_`4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ є
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_173068db_a`4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ С
6__inference_batch_normalization_3_layer_call_fn_173081Wab_`4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€АС
6__inference_batch_normalization_3_layer_call_fn_173094Wb_a`4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€Ає
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_173199dqrop4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ є
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_173222droqp4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ С
6__inference_batch_normalization_4_layer_call_fn_173235Wqrop4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€АС
6__inference_batch_normalization_4_layer_call_fn_173248Wroqp4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€АЇ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_173353eБВА3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ї
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_173376eВБА3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Т
6__inference_batch_normalization_5_layer_call_fn_173389XБВА3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "К€€€€€€€€€Т
6__inference_batch_normalization_5_layer_call_fn_173402XВБА3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "К€€€€€€€€€к
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172447Ц !"MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ к
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172469Ц !"MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ «
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172529t !"<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€В8
p
™ ".Ґ+
$К!
0€€€€€€€€€В8
Ъ «
O__inference_batch_normalization_layer_call_and_return_conditional_losses_172551t !"<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€В8
p 
™ ".Ґ+
$К!
0€€€€€€€€€В8
Ъ ¬
4__inference_batch_normalization_layer_call_fn_172482Й !"MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8¬
4__inference_batch_normalization_layer_call_fn_172495Й !"MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8Я
4__inference_batch_normalization_layer_call_fn_172564g !"<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€В8
p
™ "!К€€€€€€€€€В8Я
4__inference_batch_normalization_layer_call_fn_172577g !"<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€В8
p 
™ "!К€€€€€€€€€В8®
C__inference_flatten_layer_call_and_return_conditional_losses_172935a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€8
™ "&Ґ#
К
0€€€€€€€€€А
Ъ А
(__inference_flatten_layer_call_fn_172940T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€8
™ "К€€€€€€€€€Ао
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_169941ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_1_layer_call_fn_169947СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_170151ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_2_layer_call_fn_170157СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€м
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_169731ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_max_pooling2d_layer_call_fn_169737СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ё
J__inference_quant_conv2d_1_layer_call_and_return_conditional_losses_169787П-IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ ґ
/__inference_quant_conv2d_1_layer_call_fn_169795В-IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8ё
J__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_169997ПAIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ ґ
/__inference_quant_conv2d_2_layer_call_fn_170005ВAIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8№
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_169577ПIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ і
-__inference_quant_conv2d_layer_call_fn_169585ВIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8™
I__inference_quant_dense_1_layer_call_and_return_conditional_losses_173117]i0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ В
.__inference_quant_dense_1_layer_call_fn_173124Pi0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А©
I__inference_quant_dense_2_layer_call_and_return_conditional_losses_173271\y0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ Б
.__inference_quant_dense_2_layer_call_fn_173278Oy0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€®
G__inference_quant_dense_layer_call_and_return_conditional_losses_172963]Y0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ А
,__inference_quant_dense_layer_call_fn_172970PY0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€Аг
F__inference_sequential_layer_call_and_return_conditional_losses_171140Ш! !"-3456AGHIJYab_`iqropyБВАLҐI
BҐ?
5К2
quant_conv2d_input€€€€€€€€€В
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ г
F__inference_sequential_layer_call_and_return_conditional_losses_171221Ш! !"-3456AGHIJYb_a`iroqpyВБАLҐI
BҐ?
5К2
quant_conv2d_input€€€€€€€€€В
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ „
F__inference_sequential_layer_call_and_return_conditional_losses_172044М! !"-3456AGHIJYab_`iqropyБВА@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€В
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ „
F__inference_sequential_layer_call_and_return_conditional_losses_172271М! !"-3456AGHIJYb_a`iroqpyВБА@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€В
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ї
+__inference_sequential_layer_call_fn_171368Л! !"-3456AGHIJYab_`iqropyБВАLҐI
BҐ?
5К2
quant_conv2d_input€€€€€€€€€В
p

 
™ "К€€€€€€€€€ї
+__inference_sequential_layer_call_fn_171514Л! !"-3456AGHIJYb_a`iroqpyВБАLҐI
BҐ?
5К2
quant_conv2d_input€€€€€€€€€В
p 

 
™ "К€€€€€€€€€Ѓ
+__inference_sequential_layer_call_fn_172336! !"-3456AGHIJYab_`iqropyБВА@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€В
p

 
™ "К€€€€€€€€€Ѓ
+__inference_sequential_layer_call_fn_172401! !"-3456AGHIJYb_a`iroqpyВБА@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€В
p 

 
™ "К€€€€€€€€€б
$__inference_signature_wrapper_171733Є! !"-3456AGHIJYb_a`iroqpyВБАZҐW
Ґ 
P™M
K
quant_conv2d_input5К2
quant_conv2d_input€€€€€€€€€В"7™4
2

activation$К!

activation€€€€€€€€€†
F__inference_ste_sign_1_layer_call_and_return_conditional_losses_173441V.Ґ+
$Ґ!
К
inputs88
™ "$Ґ!
К
088
Ъ x
+__inference_ste_sign_1_layer_call_fn_173446I.Ґ+
$Ґ!
К
inputs88
™ "К88„
F__inference_ste_sign_2_layer_call_and_return_conditional_losses_173458МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ Ѓ
+__inference_ste_sign_2_layer_call_fn_173463IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8†
F__inference_ste_sign_3_layer_call_and_return_conditional_losses_173475V.Ґ+
$Ґ!
К
inputs88
™ "$Ґ!
К
088
Ъ x
+__inference_ste_sign_3_layer_call_fn_173480I.Ґ+
$Ґ!
К
inputs88
™ "К88„
F__inference_ste_sign_4_layer_call_and_return_conditional_losses_173492МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
Ъ Ѓ
+__inference_ste_sign_4_layer_call_fn_173497IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€8
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€8Ю
D__inference_ste_sign_layer_call_and_return_conditional_losses_173424V.Ґ+
$Ґ!
К
inputs8
™ "$Ґ!
К
08
Ъ v
)__inference_ste_sign_layer_call_fn_173429I.Ґ+
$Ґ!
К
inputs8
™ "К8