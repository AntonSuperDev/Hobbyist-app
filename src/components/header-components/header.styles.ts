import styled from 'styled-components/native';

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
export const OnTouch = styled.TouchableOpacity``;
