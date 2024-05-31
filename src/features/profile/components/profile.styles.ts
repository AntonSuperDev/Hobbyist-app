import {Dimensions} from 'react-native';
import styled from 'styled-components/native';

const getWidth = Dimensions.get('window').width;
const getHeight = Dimensions.get('window').height;

export const SafeView = styled.SafeAreaView`
  background-color: ${props => props.theme.colors.bg.secondary};
`;

export const ProfileContainer = styled.View`
  width: 100%;
  height: 100%;
  justify-items: center;
  padding-left: ${props => props.theme.space[6]}
  padding-right: ${props => props.theme.space[6]}
  background: rgba(255, 255, 255, 0.7);
`;
export const TierLevelIcon = styled.View`
  width: 30px;
  height: 30px;
`;
export const TierLevelText = styled.Text`
  font-weight: 500;
  position: absolute;
  left: 0;
  right: 0;
  top: 5px;
  bottom: 0;
  text-align: center;
  z-index: 1;
  color: #096c5c;
  font-size: 18px;
`;
export const UserTierText = styled.Text`
  background-color: #3CB474;
  padding-left: ${props => props.theme.space[2]}
  padding-right: ${props => props.theme.space[2]}
  color: ${props => props.theme.colors.text.inverse};
  font-size: ${props => props.theme.fontSizes.subText}
`;
export const BackgroundTop = styled.ImageBackground.attrs({
  source: require('../../../../assets/bg-top.png'),
  resizeMode: 'contain',
})`
  width: ${getWidth / 1.33}px;
  right: -5px;
  top: -5px;
  position: absolute;
  height: 147px;
`;
export const BackgroundBottom = styled.ImageBackground.attrs({
  source: require('../../../../assets/bg-bottom.png'),
  resizeMode: 'cover',
})`
  width: ${getWidth}px;
  height: ${getHeight}px;
  right: -5px;
  bottom: -5px;
  position: absolute;
  z-index: -1;
`;
export const Group = styled.View`
  justify-content: center;
  align-items: center;
`;

export const ShowPassword = styled.Pressable`
  position: absolute;
  left: 10px;
  top: 10px;
  z-index: 99;
`;
export const EyeIcon = styled.Image`
  width: 24px;
  height: 24px;
`;
export const IconDivider = styled.View`
  height: 24px;
  width: 2px;
  position: absolute;
  left: 40px;
  top: 10px;
  z-index: 9;
  border-right-style: solid;
  border-right-width: 1px;
  border-right-color: #8ecfad;
`;
export const SubmitButton = styled.View`
    width: 100%;
    align-items: center;
    border-radius:4px;
    background-color: ${props => props.theme.colors.bg.secondary}
    padding-top: ${props => props.theme.space[3]}
    padding-bottom: ${props => props.theme.space[3]}
    margin-bottom: ${props => props.theme.space[3]};
`;
export const DeleteButton = styled.View`
    width: 100%;
    align-items: center;
    border-radius:4px;
    background-color: ${props => props.theme.colors.text.error}
    padding-top: ${props => props.theme.space[3]}
    padding-bottom: ${props => props.theme.space[3]}
    margin-bottom: ${props => props.theme.space[3]};
`;
export const ButtonText = styled.Text`
  color: ${props => props.theme.colors.text.inverse};
  font-size: ${props => props.theme.fontSizes.body};
  font-weight: ${props => props.theme.fontWeights.bold};
`;

export const OnTouch = styled.TouchableOpacity`
  margin-top: 0;
  margin-bottom: 0;
`;

export const TierCollector = styled.View`
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #8ecfad;
  margin-top: ${props => props.theme.space[8]};
  padding-top: ${props => props.theme.space[4]};
`;

export const Wrapper = styled.View`
  padding-left: ${props => props.theme.space[6]};
  padding-right: ${props => props.theme.space[6]};
  padding-bottom: ${props => props.theme.space[6]};
`;

export const ProfileTitle = styled.View`
  flex-direction: row;
  justify-content: space-between;
  padding-top: ${props => props.theme.space[4]}
  padding-bottom: ${props => props.theme.space[4]}
  border-bottom-width: 1px;
  border-bottom-color: #8ecfad;
  margin-bottom: ${props => props.theme.space[4]}
`;

export const ProfileTitleText = styled.Text`
  color: ${props => props.theme.colors.text.tertiary};
  font-size: ${props => props.theme.fontSizes.body}
  font-weight: ${props => props.theme.fontWeights.bold}
`;
export const BackButton = styled.Image`
  width: 24px;
  height: 24px;
`;
export const TierHeader = styled.View`
  flex-direction: row;
  justify-content: space-between;
`;
export const TierBoxLeft = styled.View`
  flex-direction: row;
  align-items: center;
`;
export const TierImage = styled.Image`
  width: 30px;
  height: 30px;
`;
export const TierLeftText = styled.Text`
  background-color: #82BDCA;
  padding-left: ${props => props.theme.space[2]}
  padding-right: ${props => props.theme.space[2]}
  color: ${props => props.theme.colors.text.inverse};
`;
export const CurrentButton = styled.View`
  align-items: center;
  border-radius: 4px;
  background-color: ${props => props.theme.colors.bg.quinary};
  padding-top: ${props => props.theme.space[2]};
  padding-bottom: ${props => props.theme.space[2]};
  padding-left: ${props => props.theme.space[5]};
  padding-right: ${props => props.theme.space[5]};
`;
export const CurrentButtonText = styled.Text`
  color: ${props => props.theme.colors.text.inverse};
  font-size: ${props => props.theme.fontSizes.text};
  font-weight: ${props => props.theme.fontWeights.bold};
`;

export const TierCollectorList = styled.View`
  padding-top: ${props => props.theme.space[3]};
`;
export const TierListItem = styled.View`
  flex-direction: row;
  align-items: center;
  padding-bottom: ${props => props.theme.space[1]};
`;

export const ItemImage = styled.Image`
  width: 24px;
  height: 24px;
`;
export const TierListItemText = styled.Text`
  font-size: ${props => props.theme.fontSizes.subText};
  color: ${props => props.theme.colors.text.primary};
  padding-left: ${props => props.theme.space[1]};
`;
export const BackgroundBottomLeft = styled.ImageBackground.attrs({
  source: require('../../../../assets/bg-polygon1.png'),
  resizeMode: 'contain',
})`
  width: ${getWidth / 1.88}px;
  height: ${getWidth}px;
  left: 0px;
  bottom: 40px;
  position: absolute;
  z-index: -1;
`;
