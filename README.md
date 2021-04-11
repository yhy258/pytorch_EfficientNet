# pytorch_EfficientNet

Compound Scaling Methods를 통해 성능을 향상시켰습니다.  
여기에서 Compound Scaling Methods는 Network의 width, depth, resolution을 일괄적으로 scaling 해주는 것을 의미합니다.  
Compound Scaling 해주는 수치는 small grid search를 통해 찾아졌으며,  
이 Network의 baseline model은 NAS(Neural Architecture Search)를 이용해서 만들어진 네트워크 입니다.  

efficientnet은 AutoML(NAS)을 사용해서 만들어졌고, empirical한 결과입니다.
  
자세한 내용을 알고싶으시면,  
https://velog.io/@yhyj1001/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-EfficientNet-Rethinking-Model-Scaling-for-Convolutional-Neural-Networks  
를 참고해 주십시오
