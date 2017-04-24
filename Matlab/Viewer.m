function varargout = Viewer(varargin)
% VIEWER MATLAB code for Viewer.fig
%      VIEWER, by itself, creates a new VIEWER or raises the existing
%      singleton*.
%
%      H = VIEWER returns the handle to a new VIEWER or the handle to
%      the existing singleton*.
%
%      VIEWER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in VIEWER.M with the given input arguments.
%
%      VIEWER('Property','Value',...) creates a new VIEWER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Viewer_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Viewer_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Viewer

% Last Modified by GUIDE v2.5 14-Feb-2014 11:50:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Viewer_OpeningFcn, ...
                   'gui_OutputFcn',  @Viewer_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Viewer is made visible.
function Viewer_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Viewer (see VARARGIN)

% Choose default command line output for Viewer
handles.output = hObject;
if numel(varargin) ~= 2
    fprintf('Two images of equal size are needed \n');
    return;
end
set(handles.axes1,'position',[0.01, 0.01, 0.48, 0.98],'units','normalized');
set(handles.axes2,'position',[0.49, 0.01, 0.48, 0.98],'units','normalized');

handles.img1 = imagesc(varargin{1},'parent',handles.axes1);
handles.img2 = imagesc(varargin{2},'parent',handles.axes2);
handles.sz1 = size(varargin{1});
handles.sz2 = size(varargin{2});
colormap gray;

% set(handles.figure1,'units','pixels');
% pos = get(handles.figure1,'position');
% 
% handles.sz = floor([pos(3),pos(4)/2]);
% handles.canvas = zeros(handles.sz(1),handles.sz(2)*2);
% handles.X = [0 0];
% sz = size(handles.img1);
% 
% p1(1) = 1;
% p2(1) = min(size(handles.canvas,1),sz(1));
% p1(2) = 1;
% p2(2) = min(size(handles.canvas,2)/2,sz(2));
% handles.canvas(p1(1):p2(1),p1(2):p2(2)) = ...
%     mat2gray(handles.img1(p1(1):p2(1),p1(2):p2(2)));
% handles.canvas(p1(1):p2(1),(p2(2)+p1(1):p2(2)+p2(2))) = ...
%     mat2gray(handles.img2(p1(1):p2(1),p1(2):p2(2)));
%  
% fprintf('Size: [%d %d]\n',handles.sz(1), handles.sz(2));

set(handles.axes1,'Xticklabel',{}, ...
                  'Yticklabel',{}, ...
                  'ticklength',[0 0], ...
                  'dataaspectratio',[1 1 1], ...
                  'box','on');
set(handles.axes2,'Xticklabel',{}, ...
                  'Yticklabel',{}, ...
                  'ticklength',[0 0], ...
                  'dataaspectratio',[1 1 1], ...
                  'box','on');              
% Update handles structure
set(handles.figure1,'units','normalized');
guidata(hObject, handles);

% UIWAIT makes Viewer wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Viewer_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on key press with focus on figure1 or any of its controls.
function figure1_WindowKeyPressFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see FIGURE)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
% disp(eventdata.Key);


% --- Executes when figure1 is resized.
function figure1_ResizeFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function figure1_WindowButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.axes1,'units','pixels');
set(handles.axes2,'units','pixels');
if strcmp(get(handles.figure1,'SelectionType'),'normal')
    x1 = get(handles.axes1,'CurrentPoint');
    x1 = round([x1(1,1),x1(1,2)]);
%     x2 = get(handles.axes2,'CurrentPoint');
%     x2 = round([x2(1,1),x2(1,2)]);
%     if any(x2<0)
        fprintf('Zooming into [%d, %d]\n',x1(1),x1(2));
        x0 = x1(1); y0 = x1(2);
        if x0>handles.sz1(2)
            x0 = x0-handles.sz1(2);
        end
%     else
%         fprintf('second [%d, %d]\n',x2(1),x2(2));
%         x0 = x2(1); y0 = x2(2);
%     end
    h = 200;
    set(handles.axes1,'Xlim',x0+[-h, h],'Ylim',y0+[-h,h]);
    set(handles.axes2,'Xlim',x0+[-h, h],'Ylim',y0+[-h,h]);
    set(handles.axes1,'units','normalized','dataaspectratio',[1 1 1]);
    set(handles.axes2,'units','normalized','dataaspectratio',[1 1 1]);
else
    set(handles.axes1,'Xlim',[1 handles.sz1(2)],'Ylim',[1 handles.sz1(1)]);
    set(handles.axes2,'Xlim',[1 handles.sz2(2)],'Ylim',[1 handles.sz2(1)]);
    set(handles.axes1,'units','normalized','dataaspectratio',[1 1 1]);
    set(handles.axes2,'units','normalized','dataaspectratio',[1 1 1]);
end
guidata(hObject, handles);