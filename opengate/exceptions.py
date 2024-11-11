import json
import logging

ERROR_CODE_TO_HTTP_STATUS = {
    "INTERNAL_ERROR": 500,
    "INVALID_STATE": 500,
    "DATA_LOSS": 500,
    "NOT_IMPLEMENTED": 501,
    "TEMPORARILY_UNAVAILABLE": 503,
    "DEADLINE_EXCEEDED": 504,
    "REQUEST_LIMIT_EXCEEDED": 429,
    "CANCELLED": 499,
    "RESOURCE_EXHAUSTED": 429,
    "ABORTED": 409,
    "RESOURCE_CONFLICT": 409,
    "ALREADY_EXISTS": 409,
    "NOT_FOUND": 404,
    "ENDPOINT_NOT_FOUND": 404,
    "RESOURCE_DOES_NOT_EXIST": 404,
    "PERMISSION_DENIED": 403,
    "CUSTOMER_UNAUTHORIZED": 401,
    "UNAUTHENTICATED": 401,
    "BAD_REQUEST": 400,
    "RESOURCE_ALREADY_EXISTS": 400,
    "INVALID_PARAMETER_VALUE": 400,
}

HTTP_STATUS_TO_ERROR_CODE = {v: k for k, v in ERROR_CODE_TO_HTTP_STATUS.items()}
HTTP_STATUS_TO_ERROR_CODE[400] = "BAD_REQUEST"
HTTP_STATUS_TO_ERROR_CODE[404] = "ENDPOINT_NOT_FOUND"
HTTP_STATUS_TO_ERROR_CODE[500] = "INTERNAL_ERROR"

_logger = logging.getLogger(__name__)


class MlflowException(Exception):
    """
    Generic exception thrown to surface failure information about external-facing operations.
    The error message associated with this exception may be exposed to clients in HTTP responses
    for debugging purposes. If the error text is sensitive, raise a generic `Exception` object
    instead.
    """

    def __init__(self, message, error_code="INTERNAL_ERROR", **kwargs):
        """
        Args:
            message: The message or exception describing the error that occurred. This will be
                included in the exception's serialized JSON representation.
            error_code: An appropriate error code for the error that occurred; it will be
                included in the exception's serialized JSON representation. This should
                be one of the codes listed in the `mlflow.protos.databricks_pb2` proto.
            kwargs: Additional key-value pairs to include in the serialized JSON representation
                of the MlflowException.
        """
        try:
            self.error_code = error_code
        except (ValueError, TypeError):
            self.error_code = "INTERNAL_ERROR"
        message = str(message)
        self.message = message
        self.json_kwargs = kwargs
        super().__init__(message)

    def serialize_as_json(self):
        exception_dict = {"error_code": self.error_code, "message": self.message}
        exception_dict.update(self.json_kwargs)
        return json.dumps(exception_dict)

    def get_http_status_code(self):
        return ERROR_CODE_TO_HTTP_STATUS.get(self.error_code, 500)

    @classmethod
    def invalid_parameter_value(cls, message, **kwargs):
        """Constructs an `MlflowException` object with the `INVALID_PARAMETER_VALUE` error code.

        Args:
            message: The message describing the error that occurred. This will be included in the
                exception's serialized JSON representation.
            kwargs: Additional key-value pairs to include in the serialized JSON representation
                of the MlflowException.
        """
        return cls(message, error_code="INVALID_PARAMETER_VALUE", **kwargs)
