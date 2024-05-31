import styled from 'styled-components/native';
import {Dimensions} from 'react-native';

const getWidth = Dimensions.get('window').width;
const getHeight = Dimensions.get('window').height;

export const SafeView = styled.SafeAreaView`
  background-color: ${props => props.theme.colors.bg.secondary};
`;
export const Header = styled.View`
  margin-top: ${props => props.theme.space[3]}
  flex-direction: row;
  justify-content: space-between;
`;

export const Logo = styled.View``;
export const LogoImage = styled.Image`
  width: 144px;
  height: 44px;
`;
export const UserNav = styled.View`
  flex-direction: row;
`;
export const UserImage = styled.Image`
  width: 44px;
  height: 44px;
`;

export const LoginContainer = styled.View`
  width: 100%;
  height: 100%;
  justify-items: center;
  padding-left: ${props => props.theme.space[6]}
  padding-right: ${props => props.theme.space[6]}
`;

export const Welcome = styled.View`
  justify-content: center;
  align-items: center;
  margin-top: ${getWidth / 7}px;
  margin-bottom: ${getWidth / 10}px;
`;
export const WelcomeText = styled.Text`
  color: ${props => props.theme.colors.text.inverse};
  font-size: ${props => props.theme.fontSizes.h5};
  font-weight: ${props => props.theme.fontWeights.bold};
  text-transform: uppercase;
  margin-bottom: ${props => props.theme.space[3]};
`;

export const WelcomeSloganText = styled.Text`
  color: ${props => props.theme.colors.text.inverse};
  font-size: ${props => props.theme.fontSizes.text};
  text-align: center;
`;

export const OnTouchSignUp = styled.TouchableOpacity``;
export const SignUpText = styled.Text`
  color: ${props => props.theme.colors.text.quaternary};
  font-size: ${props => props.theme.fontSizes.text};
  font-weight: ${props => props.theme.fontWeights.bold};
  text-decoration: underline;
  text-decoration-color: ${props => props.theme.colors.text.quaternary};
  margin-bottom: -4px;
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
export const SubmitButton = styled.ImageBackground.attrs({
  source: require('../../../../assets/bg-button.png'),
  resizeMode: 'cover',
  width: '100%',
  borderRadius: 4,
})`
    width: 100%;
    align-items: center;
    border-radius:4px;
    padding-top: ${props => props.theme.space[3]}
    padding-bottom: ${props => props.theme.space[3]}
    margin-bottom: ${props => props.theme.space[3]};
`;
export const ButtonText = styled.Text`
  color: ${props => props.theme.colors.text.secondary};
  font-size: ${props => props.theme.fontSizes.body};
  font-weight: ${props => props.theme.fontWeights.bold};
`;

export const OnTouch = styled.TouchableOpacity`
  margin-top: 0;
  margin-bottom: 0;
`;

export const TermAndContions = styled.View`
  border-bottom-style: solid;
  border-bottom-width: 1px;
  border-bottom-color: #8ecfad;
  margin-bottom: ${props => props.theme.space[8]};
`;
export const TermAndContionsText = styled.Text`
  color: ${props => props.theme.colors.text.inverse};
  font-size: ${props => props.theme.fontSizes.subCaption};
  text-align: center;
  margin-bottom: 14px;
`;
export const TermAndContionsTextColor = styled.Text`
  color: ${props => props.theme.colors.text.quaternary};
  font-size: ${props => props.theme.fontSizes.subCaption};
  text-decoration: underline;
  text-decoration-color: ${props => props.theme.colors.text.quaternary};
`;
export const Loading = styled.ActivityIndicator``;
