import styled from 'styled-components/native';
import {Dimensions} from 'react-native';

const getWidth = Dimensions.get('window').width;
const getHeight = Dimensions.get('window').height;

export const BackgroundTop = styled.ImageBackground.attrs({
  source: require('../../../assets/bg-top.png'),
  resizeMode: 'contain',
})`
  width: ${getWidth / 1.33}px;
  right: -5px;
  top: -5px;
  position: absolute;
  height: 147px;
`;
export const BackgroundBottom = styled.ImageBackground.attrs({
  source: require('../../../assets/bg-bottom.png'),
  resizeMode: 'cover',
})`
  width: ${getWidth}px;
  height: ${getHeight}px;
  right: -5px;
  bottom: -5px;
  position: absolute;
  z-index: -1;
`;
export const BackgroundBottomLeft = styled.ImageBackground.attrs({
  source: require('../../../assets/bg-polygon1.png'),
  resizeMode: 'contain',
})`
  width: ${getWidth / 1.88}px;
  height: ${getWidth}px;
  left: 0px;
  bottom: 30px;
  position: absolute;
  z-index: -1;
`;
