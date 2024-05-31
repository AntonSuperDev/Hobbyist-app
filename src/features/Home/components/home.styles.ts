import {Dimensions, FlatList} from 'react-native';

import styled from 'styled-components/native';

const getWidth = Dimensions.get('window').width;
const getHeight = Dimensions.get('window').height;

export const Wrapper = styled.View`
  padding-left: ${props => props.theme.space[6]};
  padding-right: ${props => props.theme.space[6]};
`;

export const SearchContainer = styled.View`
  margin-top: ${props => props.theme.space[6]};
  margin-bottom: ${props => props.theme.space[6]};
  padding-left: ${props => props.theme.space[6]};
  padding-right: ${props => props.theme.space[6]};
  flex-direction: row;
`;
export const SearchInput = styled.TextInput.attrs({
  placeholderTextColor: '#096C5C',
})`
  height: ${props => props.theme.sizes[5]};
  width: 100%;
  background: rgba(255, 255, 255, 0.56);
  font-weight: bold;
  font-size: 13px;
  padding-left: ${props => props.theme.space[3]};
  padding-right: ${props => props.theme.space[3]};
  padding-top: ${props => props.theme.space[2]}
  padding-bottom: ${props => props.theme.space[2]}
  text-transform: uppercase;
  border-top-left-radius: 4px;
  border-bottom-left-radius: 4px;
`;
export const SelectInput = styled.TextInput.attrs({
  placeholderTextColor: '#096C5C',
})`
  height: ${props => props.theme.sizes[5]};
  width: 100%;
  background: rgba(255, 255, 255, 0.56);
  font-weight: bold;
  font-size: 13px;
  padding-left: ${props => props.theme.space[3]};
  padding-right: ${props => props.theme.space[3]};
  padding-top: ${props => props.theme.space[2]}
  padding-bottom: ${props => props.theme.space[2]}
  border-top-right-radius: 4px;
  border-bottom-right-radius: 4px;
  
`;
// width: ${getWidth - 48 - (getWidth - 48) / 3}px;
export const SearchBox = styled.View`
  width: ${getWidth}px;
`;
export const InputDropdown = styled.View`
  width: ${(getWidth - 48) / 3}px;
`;
export const ImageIcon = styled.Image`
  width: 15px;
  height: 15px;
  position: absolute;
  right: 20px;
  top: 15px;
`;
export const IconDivider = styled.View`
  height: 24px;
  width: 2px;
  position: absolute;
  left: 0px;
  top: 10px;
  z-index: 9;
  border-right-style: solid;
  border-right-width: 1px;
  border-right-color: #096c5c;
`;

export const FlatListWrapper = styled.View`
  min-height: 60%;
  background: rgba(255, 255, 255, 0.9);
`;
export const BackgroundPolygon = styled.ImageBackground.attrs({
  source: require('../../../../assets/bg-polygon.png'),
  resizeMode: 'cover',
})`
  width: ${getWidth}px;
  height: ${getHeight / 2.5}px;
  left: -5px;
  top: 0px;
  position: absolute;
  z-index: -1;
`;
export const FlatListView = styled(FlatList).attrs({
  paddingLeft: 16,
  paddingRight: 16,
  marginTop: 16,
  marginBottom: 60,
  // height: getHeight * 0.6,
})``;

export const NoItem = styled.Text`
  color: ${props => props.theme.colors.text.dark};
  font-size: ${props => props.theme.fontSizes.caption};
  font-weight: ${props => props.theme.fontWeights.bold};
  margin-bottom: ${props => props.theme.space[3]};
  align-self: center;
`;