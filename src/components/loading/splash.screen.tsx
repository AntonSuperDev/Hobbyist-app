import React, {FC} from 'react';
import styled from 'styled-components/native';

const SplashImage = styled.ImageBackground.attrs({
  source: require('../../../assets/splash.png'),
  resizeMode: 'contain',
  backgroundPosition: 'center',
})`
  flex: 1;
  justify-content: center;
  align-items: center;
  background-color: #3cb474;
`;

const SplashScreen: FC = () => <SplashImage />;

export default SplashScreen;
