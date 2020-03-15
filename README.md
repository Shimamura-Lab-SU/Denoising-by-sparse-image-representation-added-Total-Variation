# Denoising-by-sparse-image-representation-added-Total-Variation

## proposal_method_lambda.m(初回のみ)
雑音画像の生成  
l.69,107のコメントアウトを外し,下のimportdataの行をコメントアウトして実行

## Dic_Noiseimg_generater.m(初回のみ)
雑音画像から辞書を生成

## proposal_method_lambda.m
雑音画像に対しスパースのみを考慮して雑音除去したときのスパース正則化項の係数lambdaの最適値と結果画像を出力

## proposal_method_theta.m
proposal_method_lambda.mで求めた最適値にlambdaを固定しTV正則化項の係数thetaを動かしその最適地と結果画像を出力
