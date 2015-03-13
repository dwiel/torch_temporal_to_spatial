All of the following timings are run on amazon ec2 GPU instance.


nn.SpatialConvolutionNN:cuda is a using the nn.SpatialConvolutionNN
module in a raw way, no transforms on the input or output are done to
make 1d data appear 2d

TemporalConvolutionMM:cuda is implemented in this repo and wrapps
nn.SpatialConvolutionNN with reshapes, transposes and views to
reorganize 1d data to appear 2d

nn.SpatialConvolutionMM is the same as nn.SpatialConvolutionNN:cuda
but run on the CPU

TemporalConvolutionMM is the same as TemporalConvolutionMM:cuda but
run on the CPU

TemporalConvolution is similar to TemporalConvolutionMM but is
wrapping SpatialConvolution instead of SpatialConvolutionMM

nn.TemporalConvolution is the standard implementation provided in nn
package


running `th benchmark.lua` returns this:
```
Running on device: GRID K520	
	
CONFIG: input = 8640 * ker = 1x64x6 (bs = 128, stride = 6)	
nn.SpatialConvolutionMM:cuda       forward():          (tm = 0.0051684975624084)	
TemporalConvolutionMM:cuda         forward():          (tm = 0.0097372531890869)	
nn.SpatialConvolutionMM            forward():          (tm = 0.0097077488899231)	
TemporalConvolutionMM              forward():          (tm = 0.11057752370834)	
TemporalConvolution                forward():          (tm = 0.12648624181747)	
nn.TemporalConvolution             forward():          (tm = 0.086997032165527)	
	
nn.SpatialConvolutionMM:cuda       outputOutput():     (tm = 0.0050640106201172)	
TemporalConvolutionMM:cuda         outputOutput():     (tm = 0.0096734762191772)	
nn.SpatialConvolutionMM            outputOutput():     (tm = 0.0078295469284058)	
TemporalConvolutionMM              outputOutput():     (tm = 0.10873347520828)	
TemporalConvolution                outputOutput():     (tm = 0.12361252307892)	
nn.TemporalConvolution             outputOutput():     (tm = 0.085354745388031)	
	
nn.SpatialConvolutionMM:cuda       updateGradient():   (tm = 0.0040780305862427)	
TemporalConvolutionMM:cuda         updateGradient():   (tm = 0.0088972449302673)	
nn.SpatialConvolutionMM            updateGradient():   (tm = 0.0086544752120972)	
TemporalConvolutionMM              updateGradient():   (tm = 0.10782778263092)	
TemporalConvolution                updateGradient():   (tm = 0.12077629566193)	
nn.TemporalConvolution             updateGradient():   (tm = 0.024791240692139)	
	
nn.SpatialConvolutionMM:cuda       accGradParameters():(tm = 0.017757713794708)	
TemporalConvolutionMM:cuda         accGradParameters():(tm = 0.017813742160797)	
nn.SpatialConvolutionMM            accGradParameters():(tm = 0.031462490558624)	
TemporalConvolutionMM              accGradParameters():(tm = 0.03146880865097)	
TemporalConvolution                accGradParameters():(tm = 0.02506422996521)	
nn.TemporalConvolution             accGradParameters():(tm = 0.052875280380249)	
	
	
CONFIG: input = 1440 * ker = 64x64x6 (bs = 128, stride = 1)	
nn.SpatialConvolutionMM:cuda       forward():          (tm = 0.021595239639282)	
TemporalConvolutionMM:cuda         forward():          (tm = 0.030954778194427)	
nn.SpatialConvolutionMM            forward():          (tm = 0.17447948455811)	
TemporalConvolutionMM              forward():          (tm = 0.42530977725983)	
TemporalConvolution                forward():          (tm = 0.87654852867126)	
nn.TemporalConvolution             forward():          (tm = 0.23711770772934)	
	
nn.SpatialConvolutionMM:cuda       outputOutput():     (tm = 0.021579027175903)	
TemporalConvolutionMM:cuda         outputOutput():     (tm = 0.03094869852066)	
nn.SpatialConvolutionMM            outputOutput():     (tm = 0.1752587556839)	
TemporalConvolutionMM              outputOutput():     (tm = 0.41839325428009)	
TemporalConvolution                outputOutput():     (tm = 0.87786275148392)	
nn.TemporalConvolution             outputOutput():     (tm = 0.23293429613113)	
	
nn.SpatialConvolutionMM:cuda       updateGradient():   (tm = 0.082946300506592)	
TemporalConvolutionMM:cuda         updateGradient():   (tm = 0.092243194580078)	
nn.SpatialConvolutionMM            updateGradient():   (tm = 0.1648765206337)	
TemporalConvolutionMM              updateGradient():   (tm = 0.3661299943924)	
TemporalConvolution                updateGradient():   (tm = 0.82629150152206)	
nn.TemporalConvolution             updateGradient():   (tm = 0.16675353050232)	
	
nn.SpatialConvolutionMM:cuda       accGradParameters():(tm = 0.057855725288391)	
TemporalConvolutionMM:cuda         accGradParameters():(tm = 0.057813227176666)	
nn.SpatialConvolutionMM            accGradParameters():(tm = 0.14934176206589)	
TemporalConvolutionMM              accGradParameters():(tm = 0.15142351388931)	
TemporalConvolution                accGradParameters():(tm = 1.1790764927864)	
nn.TemporalConvolution             accGradParameters():(tm = 0.20846450328827)	
	
	
CONFIG: input = 20160 * ker = 64x64x6 (bs = 16, stride = 1)	
nn.SpatialConvolutionMM:cuda       forward():          (tm = 0.031190752983093)	
TemporalConvolutionMM:cuda         forward():          (tm = 0.057639539241791)	
nn.SpatialConvolutionMM            forward():          (tm = 0.31597876548767)	
TemporalConvolutionMM              forward():          (tm = 0.70255649089813)	
TemporalConvolution                forward():          (tm = 1.9500250220299)	
nn.TemporalConvolution             forward():          (tm = 0.33523726463318)	
	
nn.SpatialConvolutionMM:cuda       outputOutput():     (tm = 0.031095027923584)	
TemporalConvolutionMM:cuda         outputOutput():     (tm = 0.057571470737457)	
nn.SpatialConvolutionMM            outputOutput():     (tm = 0.30649524927139)	
TemporalConvolutionMM              outputOutput():     (tm = 0.6957945227623)	
TemporalConvolution                outputOutput():     (tm = 1.9456139802933)	
nn.TemporalConvolution             outputOutput():     (tm = 0.32961624860764)	
	
nn.SpatialConvolutionMM:cuda       updateGradient():   (tm = 0.039806962013245)	
TemporalConvolutionMM:cuda         updateGradient():   (tm = 0.066052973270416)	
nn.SpatialConvolutionMM            updateGradient():   (tm = 0.30233126878738)	
TemporalConvolutionMM              updateGradient():   (tm = 0.65497797727585)	
TemporalConvolution                updateGradient():   (tm = 1.9136654734612)	
nn.TemporalConvolution             updateGradient():   (tm = 0.23770099878311)	
	
nn.SpatialConvolutionMM:cuda       accGradParameters():(tm = 0.10011446475983)	
TemporalConvolutionMM:cuda         accGradParameters():(tm = 0.1003640294075)	
nn.SpatialConvolutionMM            accGradParameters():(tm = 0.25763672590256)	
TemporalConvolutionMM              accGradParameters():(tm = 0.25709849596024)	
TemporalConvolution                accGradParameters():(tm = 2.0568840503693)	
nn.TemporalConvolution             accGradParameters():(tm = 0.31178897619247)	
	
	
CONFIG: input = 32 * ker = 128x128x9 (bs = 128, stride = 1)	
nn.SpatialConvolutionMM:cuda       forward():          (tm = 0.017311275005341)	
TemporalConvolutionMM:cuda         forward():          (tm = 0.017542243003845)	
nn.SpatialConvolutionMM            forward():          (tm = 0.015855550765991)	
TemporalConvolutionMM              forward():          (tm = 0.022887706756592)	
TemporalConvolution                forward():          (tm = 0.091885268688202)	
nn.TemporalConvolution             forward():          (tm = 0.32191449403763)	
	
nn.SpatialConvolutionMM:cuda       outputOutput():     (tm = 0.017270267009735)	
TemporalConvolutionMM:cuda         outputOutput():     (tm = 0.017517745494843)	
nn.SpatialConvolutionMM            outputOutput():     (tm = 0.015424787998199)	
TemporalConvolutionMM              outputOutput():     (tm = 0.022409737110138)	
TemporalConvolution                outputOutput():     (tm = 0.092250227928162)	
nn.TemporalConvolution             outputOutput():     (tm = 0.32643747329712)	
	
nn.SpatialConvolutionMM:cuda       updateGradient():   (tm = 0.0073364973068237)	
TemporalConvolutionMM:cuda         updateGradient():   (tm = 0.0076937675476074)	
nn.SpatialConvolutionMM            updateGradient():   (tm = 0.016064465045929)	
TemporalConvolutionMM              updateGradient():   (tm = 0.023745238780975)	
TemporalConvolution                updateGradient():   (tm = 0.086959779262543)	
nn.TemporalConvolution             updateGradient():   (tm = 0.2989040017128)	
	
nn.SpatialConvolutionMM:cuda       accGradParameters():(tm = 0.010874271392822)	
TemporalConvolutionMM:cuda         accGradParameters():(tm = 0.010811507701874)	
nn.SpatialConvolutionMM            accGradParameters():(tm = 0.01622873544693)	
TemporalConvolutionMM              accGradParameters():(tm = 0.017320036888123)	
TemporalConvolution                accGradParameters():(tm = 0.13061529397964)	
nn.TemporalConvolution             accGradParameters():(tm = 0.048566997051239)	
	
	
CONFIG: input = 16 * ker = 128x128x7 (bs = 128, stride = 1)	
nn.SpatialConvolutionMM:cuda       forward():          (tm = 0.010449230670929)	
TemporalConvolutionMM:cuda         forward():          (tm = 0.010503232479095)	
nn.SpatialConvolutionMM            forward():          (tm = 0.0071084499359131)	
TemporalConvolutionMM              forward():          (tm = 0.010894775390625)	
TemporalConvolution                forward():          (tm = 0.045309484004974)	
nn.TemporalConvolution             forward():          (tm = 0.17057597637177)	
	
nn.SpatialConvolutionMM:cuda       outputOutput():     (tm = 0.010362029075623)	
TemporalConvolutionMM:cuda         outputOutput():     (tm = 0.010412514209747)	
nn.SpatialConvolutionMM            outputOutput():     (tm = 0.0069580078125)	
TemporalConvolutionMM              outputOutput():     (tm = 0.010591506958008)	
TemporalConvolution                outputOutput():     (tm = 0.045007288455963)	
nn.TemporalConvolution             outputOutput():     (tm = 0.16968375444412)	
	
nn.SpatialConvolutionMM:cuda       updateGradient():   (tm = 0.005715012550354)	
TemporalConvolutionMM:cuda         updateGradient():   (tm = 0.0058124661445618)	
nn.SpatialConvolutionMM            updateGradient():   (tm = 0.0074484944343567)	
TemporalConvolutionMM              updateGradient():   (tm = 0.011077761650085)	
TemporalConvolution                updateGradient():   (tm = 0.037640273571014)	
nn.TemporalConvolution             updateGradient():   (tm = 0.14999997615814)	
	
nn.SpatialConvolutionMM:cuda       accGradParameters():(tm = 0.0057199597358704)	
TemporalConvolutionMM:cuda         accGradParameters():(tm = 0.0057334899902344)	
nn.SpatialConvolutionMM            accGradParameters():(tm = 0.0084140300750732)	
TemporalConvolutionMM              accGradParameters():(tm = 0.0089792609214783)	
TemporalConvolution                accGradParameters():(tm = 0.055418252944946)	
nn.TemporalConvolution             accGradParameters():(tm = 0.066212475299835)	
	
	
CONFIG: input = 13 * ker = 384x384x3 (bs = 128, stride = 1)	
nn.SpatialConvolutionMM:cuda       forward():          (tm = 0.017492234706879)	
TemporalConvolutionMM:cuda         forward():          (tm = 0.017699539661407)	
nn.SpatialConvolutionMM            forward():          (tm = 0.02733850479126)	
TemporalConvolutionMM              forward():          (tm = 0.036476790904999)	
TemporalConvolution                forward():          (tm = 0.28784501552582)	
nn.TemporalConvolution             forward():          (tm = 0.30812919139862)	
	
nn.SpatialConvolutionMM:cuda       outputOutput():     (tm = 0.017441511154175)	
TemporalConvolutionMM:cuda         outputOutput():     (tm = 0.01758348941803)	
nn.SpatialConvolutionMM            outputOutput():     (tm = 0.027449727058411)	
TemporalConvolutionMM              outputOutput():     (tm = 0.035983741283417)	
TemporalConvolution                outputOutput():     (tm = 0.28676396608353)	
nn.TemporalConvolution             outputOutput():     (tm = 0.3079394698143)	
	
nn.SpatialConvolutionMM:cuda       updateGradient():   (tm = 0.013721466064453)	
TemporalConvolutionMM:cuda         updateGradient():   (tm = 0.013884007930756)	
nn.SpatialConvolutionMM            updateGradient():   (tm = 0.029807269573212)	
TemporalConvolutionMM              updateGradient():   (tm = 0.039348721504211)	
TemporalConvolution                updateGradient():   (tm = 0.24696224927902)	
nn.TemporalConvolution             updateGradient():   (tm = 0.26542901992798)	
	
nn.SpatialConvolutionMM:cuda       accGradParameters():(tm = 0.01530796289444)	
TemporalConvolutionMM:cuda         accGradParameters():(tm = 0.015299260616302)	
nn.SpatialConvolutionMM            accGradParameters():(tm = 0.027607262134552)	
TemporalConvolutionMM              accGradParameters():(tm = 0.029064238071442)	
TemporalConvolution                accGradParameters():(tm = 0.40906572341919)	
nn.TemporalConvolution             accGradParameters():(tm = 0.054712951183319)	
```